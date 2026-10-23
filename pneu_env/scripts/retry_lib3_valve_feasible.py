"""Single-valve feasibility-first calibration pilot (type1, v2).

Preserves the original tuning workspace and deployed simulator. The fitted
hysteresis requires sim_core_feasible.cpp; exporting its constants alone into
the old Euler-z implementation is NOT a supported deployment path.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


POINTER = ctypes.POINTER(ctypes.c_double)
ROOT = Path(__file__).resolve().parents[2]
TUNE_ROOT = ROOT.parent / "pneu_tune"


def plot_surface(trace, parameters, destination):
    """v2 only: atmospheric upstream, measured downstream, z-normalized h=0 surface."""
    p = parameters
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    pressure, command = np.meshgrid(np.linspace(trace.pout.min(), trace.pout.max(), 40),
                                    np.linspace(.8, 1, 40))
    ratio = pressure/101.325
    critical = (2/2.4)**(1.4/.4)
    phi = np.where(ratio <= critical, np.sqrt(1.4*(2/2.4)**(2.4/.4)),
                   np.sqrt(np.maximum(0, 7*(ratio**(2/1.4)-ratio**(2.4/1.4)))))
    area = np.exp(-p[8]*np.logaddexp(0, -p[1]*(.6*command-.3-p[2])))
    ax.plot_surface(pressure, command, p[0]*area*101.325*phi, alpha=.2, color="steelblue")
    sampled = trace.iloc[::max(1, len(trace)//2000)]
    ax.scatter(sampled.pout, sampled.ctrl, sampled.real, c="black", s=4, label="Real")
    ax.scatter(sampled.pout, sampled.ctrl, sampled.candidate, c="tomato", s=4, label="Pilot dynamic")
    ax.set(xlabel="Downstream pressure (kPa)", ylabel="Physical command", zlabel="Flow (L/min)",
           title="v2: measured / dynamic prediction / static h=0 surface")
    ax.legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def build_core():
    source = Path(__file__).resolve().parents[1]/"src/pneu_env/tuner/sim_core_feasible.cpp"
    library = Path(tempfile.mkdtemp(prefix="lib3_feasible_v2_"))/"sim_core_feasible.so"
    subprocess.run(["g++", "-O3", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    core = ctypes.CDLL(str(library))
    core.simulate_feasible.argtypes = [ctypes.c_int] + [POINTER]*5 + [ctypes.c_double] + [POINTER]*5
    core.simulate_feasible.restype = ctypes.c_int
    return core, source, library


def make_anchors(data, reference):
    """Recorded settled-tail envelopes; does not assume u=.8 means closed."""
    u = .5+.5*data["u"]
    t = data["Time"]
    cuts = np.r_[0, np.flatnonzero(abs(np.diff(u)) > 1e-4)+1, len(u)]
    anchors = []
    tolerance = .05*reference  # Declared pilot acceptance allowance, not measured sensor accuracy.
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if t[stop-1]-t[start]+data["dt"] < 1.5:
            continue
        ids = np.arange(start, stop)
        ids = ids[t[ids] >= t[stop-1]-.5]
        if len(ids) < 10:
            continue
        q = data["Q"][ids]
        anchors.append(dict(kind="hold_tail", ctrl=float(np.median(u[ids])),
                            start=float(t[start]), end=float(t[stop-1]), ids=ids,
                            lo=float(np.quantile(q, .05)-tolerance),
                            hi=float(np.quantile(q, .95)+tolerance),
                            real_mean=float(q.mean()),
                            branch=int(data["State"][ids[-1]]),
                            pin=float(np.median(data["P_in_abs"][ids])),
                            pout=float(np.median(data["P_out_abs"][ids]))))
    # Full-open pressure bands, using only samples held >=1s.
    changed = np.r_[True, abs(np.diff(u)) > 1e-4]
    last = np.maximum.accumulate(np.where(changed, np.arange(len(u)), 0))
    age = (np.arange(len(u))-last)*data["dt"]
    for lo in range(20, 50, 5):
        ids = np.flatnonzero((u > .999) & (age >= 1)
                            & (data["P_out_abs"] >= lo) & (data["P_out_abs"] < lo+5))
        if len(ids) < 25:
            continue
        q = data["Q"][ids]
        anchors.append(dict(kind="full_open_pressure_band", ctrl=1., start=float(t[ids[0]]),
                            end=float(t[ids[-1]]), ids=ids,
                            lo=float(np.quantile(q, .05)-tolerance),
                            hi=float(np.quantile(q, .95)+tolerance),
                            real_mean=float(q.mean()), branch=-1,
                            pin=float(np.median(data["P_in_abs"][ids])),
                            pout=float(np.median(data["P_out_abs"][ids]))))
    return anchors


class Pilot:
    def __init__(self, core, data, anchors, reference, wn_bounds=(11., 25.), zeta_bounds=(.4, .6)):
        self.core, self.data, self.anchors, self.reference = core, data, anchors, reference
        self.arrays = [np.ascontiguousarray(data[key], dtype=np.float64)
                       for key in ("I", "State", "P_in_abs", "Phi", "Q")]
        self.ptrs = [a.ctypes.data_as(POINTER) for a in self.arrays]
        self.buffers = [np.empty(len(data["Q"])) for _ in range(4)]
        self.outputs = [a.ctypes.data_as(POINTER) for a in self.buffers]
        self.capacity = float(np.median(data["P_in_abs"]*data["Phi"]))
        self.last_x = None
        self.calls = 0
        self.best_feasible = None
        self.wn_bounds, self.zeta_bounds = wn_bounds, zeta_bounds

    def parameters(self, x):
        # Normalize identifiable parameters. v2 Pin is constant: Cp and Ck
        # cannot be separately identified, so fit effective Ck with Cp=0.
        qmax = self.reference*(.85+.4*x[0])
        k = np.exp(np.log(5)+x[1]*np.log(200/5))
        alpha = np.exp(np.log(.01)+x[2]*np.log(20000/.01))
        u50 = .70+.35*x[3]
        force50 = -np.log(np.expm1(np.log(2)/alpha))/k
        ck = (.6*u50-.3)-force50
        amplitude = .06*(2*x[4]-1)
        rate = np.exp(x[5]*np.log(2000))
        fraction = .5+.5*x[6]
        wnlo, wnhi = self.wn_bounds
        zlo, zhi = self.zeta_bounds
        return np.array([qmax/self.capacity, k, ck, 0., amplitude, rate,
                         rate*fraction, rate*(1-fraction), alpha,
                         wnlo+(wnhi-wnlo)*x[7], zlo+(zhi-zlo)*x[8],
                         wnlo+(wnhi-wnlo)*x[9], zlo+(zhi-zlo)*x[10]])

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.value, self.margins
        self.calls += 1
        self.last_x = np.array(x, copy=True)
        p = self.parameters(x)
        if p[2] < 0 or not np.isfinite(p).all():
            self.value = 1e3+max(0, -p[2])*100
            self.margins = np.full(2*len(self.anchors)+6, -1.)
            return self.value, self.margins
        status = self.core.simulate_feasible(len(self.arrays[0]), *self.ptrs, self.data["dt"],
                                             p.ctypes.data_as(POINTER), *self.outputs)
        if status:
            self.value = 1e3+status
            self.margins = np.full(2*len(self.anchors)+6, -1.)
            return self.value, self.margins
        pred, h, area, force = self.buffers
        margins = []
        for anchor in self.anchors:
            mean = float(np.mean(pred[anchor["ids"]]))
            margins.extend([(mean-anchor["lo"])/self.reference,
                            (anchor["hi"]-mean)/self.reference])
        # Explicit full-opening convention at normalized histories -1,0,+1.
        # Pin fixed in this dataset. These are model priors, not observed area.
        ld = p[8]*np.logaddexp(0, -p[1]*(.3+p[4]*np.array([-1., 0., 1.])-p[2]))
        margins.extend(np.exp(-ld)-.95)
        # Absolute trace envelope derived from measured percentiles; no mean-based startup gate.
        margins.extend([(pred.min()+.01*self.reference)/self.reference,
                        (1.2*np.quantile(self.data["Q"], .999)-pred.max())/self.reference,
                        (1+1e-10-np.max(abs(h)))])
        self.margins = np.asarray(margins)
        self.value = float(np.mean((pred-self.data["Q"])**2)/self.reference**2)
        if self.margins.min() >= -1e-6:
            if self.best_feasible is None or self.value < self.best_feasible[0]:
                self.best_feasible = (self.value, np.array(x, copy=True))
        return self.value, self.margins

    def screening(self, x):
        loss, margins = self.evaluate(x)
        return loss+1000*np.sum(np.minimum(margins, 0)**2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-root", type=Path, default=TUNE_ROOT)
    ap.add_argument("--output-dir", type=Path, default=TUNE_ROOT / "exp/lib3_v2_feasible_retry_20261022")
    ap.add_argument("--trace-root", type=Path, default=TUNE_ROOT / "exp/lib3_v2_feasible_retry_20261022/traces")
    ap.add_argument("--de-iterations", type=int, default=45)
    ap.add_argument("--local-iterations", type=int, default=250)
    ap.add_argument("--warm-start", type=Path, help="Optional pilot NPZ to seed another search of this valve")
    ap.add_argument("--wn-min", type=float, default=11.)
    ap.add_argument("--wn-max", type=float, default=25.)
    ap.add_argument("--zeta-min", type=float, default=.4)
    ap.add_argument("--zeta-max", type=float, default=.6)
    args = ap.parse_args()
    if not (0 < args.wn_min < args.wn_max and 0 < args.zeta_min < args.zeta_max):
        ap.error("Require positive, increasing wn and zeta bounds")
    out, root = args.output_dir, args.tune_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sys.path.insert(0, str(root/"src"))
    optimizer = root/"src/pneu_tune/optimizer.py"
    spec = importlib.util.spec_from_file_location("original_valve_optimizer", optimizer)
    opt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt)
    raw = next((root/"exp").glob("261022*type1*.csv"))
    data = opt.load_and_preprocess(str(raw), opt.CONFIGS[1], start_time_sec=0,
                                   input_window_size=1, flow_window_size=1)
    reference = float(np.median(data["Q"][data["u"] > .999]))
    anchors = make_anchors(data, reference)
    assert any(a["kind"] == "hold_tail" and a["ctrl"] < .96 for a in anchors)
    core, source, library = build_core()
    pilot = Pilot(core, data, anchors, reference, (args.wn_min, args.wn_max),
                  (args.zeta_min, args.zeta_max))
    # Same data, deterministic seed. Global search finds an admissible basin;
    # final SLSQP minimizes normalized SSE subject to explicit inequalities.
    progress = 0
    def callback(x, convergence):
        nonlocal progress
        progress += 1
        if progress % 10 == 0:
            loss, g = pilot.evaluate(x)
            print(f"screen {progress}: RMSE={np.sqrt(loss)*reference:.4f}, min_margin={g.min():.5f}, feasible_seen={pilot.best_feasible is not None}", flush=True)
    initial_population = "latinhypercube"
    if args.warm_start:
        center = np.load(args.warm_start)["normalized_x"]
        prior = np.load(args.warm_start)["parameters"]
        center[[7, 9]] = np.clip((prior[[9, 11]]-args.wn_min)/(args.wn_max-args.wn_min), 0, 1)
        center[[8, 10]] = np.clip((prior[[10, 12]]-args.zeta_min)/(args.zeta_max-args.zeta_min), 0, 1)
        rng = np.random.default_rng(0)
        initial_population = np.clip(center+rng.normal(0, .15, (88, 11)), 0, 1)
        initial_population[0] = center
        # Explicitly explore both signs of a bounded hysteresis contribution.
        for j, amplitude in enumerate((-.02, -.01, -.005, .005, .01, .02), start=1):
            initial_population[j] = center
            initial_population[j, 4] = (amplitude/.06+1)/2
    global_result = differential_evolution(pilot.screening, [(0, 1)]*11,
                                           maxiter=args.de_iterations, popsize=8,
                                           seed=0, polish=False, callback=callback,
                                           tol=1e-5, init=initial_population)
    starts = [global_result.x]
    for amplitude in (-.01, .01):
        alternative = global_result.x.copy()
        alternative[4] = (amplitude/.06+1)/2
        starts.append(alternative)
    if pilot.best_feasible is not None:
        starts.append(pilot.best_feasible[1])
    local_results = []
    for index, initial in enumerate(starts):
        result = minimize(lambda x: pilot.evaluate(x)[0], initial, method="SLSQP",
                          bounds=[(0, 1)]*11,
                          constraints=[dict(type="ineq", fun=lambda x: pilot.evaluate(x)[1])],
                          options=dict(maxiter=args.local_iterations, ftol=1e-10))
        loss, g = pilot.evaluate(result.x)
        local_results.append(dict(success=bool(result.success), message=str(result.message),
                                  iterations=int(result.nit), rmse=np.sqrt(loss)*reference,
                                  min_margin=float(g.min())))
        print(f"local {index+1}: {local_results[-1]}", flush=True)
    accepted = pilot.best_feasible is not None
    x = pilot.best_feasible[1] if accepted else global_result.x
    # Invalidate cache so all output buffers match the selected candidate.
    pilot.last_x = None
    loss, g = pilot.evaluate(x)
    p = pilot.parameters(x)
    pred, h, area, force = [a.copy() for a in pilot.buffers]
    import re
    baseline_path = root/"tune_result/output_261022_type1_baseline.txt"
    baseline_text = baseline_path.read_text()
    block = re.search(r"CHAMBER_NEG_PARAMS\s*=\s*\{([^}]+)", baseline_text)[1]
    baseline_p = np.array([float(v) for v in block.split(",") if v.strip()])
    _, baseline_q = opt.simulate_physics_model(data, baseline_p)
    def metrics(q):
        e = q-data["Q"]
        return dict(rmse_lpm=float(np.sqrt(np.mean(e**2))),
                    r2=float(1-np.sum(e**2)/np.sum((data["Q"]-np.mean(data["Q"]))**2)),
                    hf_ratio=float(np.std(np.diff(q))/np.std(np.diff(data["Q"]))),
                    q_min=float(q.min()), q_max=float(q.max()))
    anchor_rows = []
    for a in anchors:
        ids = a["ids"]
        row = {k: v for k, v in a.items() if k != "ids"}
        row.update(pred_mean=float(pred[ids].mean()), baseline_mean=float(baseline_q[ids].mean()))
        row["pass"] = bool(row["lo"]-1e-5 <= row["pred_mean"] <= row["hi"]+1e-5)
        anchor_rows.append(row)
    pd.DataFrame(anchor_rows).to_csv(out/"anchors.csv", index=False)
    trace_dir = args.trace_root
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir/"replay.csv"
    pd.DataFrame(dict(time=data["Time"], ctrl=.5+.5*data["u"], real=data["Q"],
                      baseline=baseline_q, candidate=pred, h=h, area=area, force=force,
                      pin=data["P_in_abs"], pout=data["P_out_abs"])).to_csv(trace_path, index=False)
    np.savez(out/"parameters.npz", parameters=p, normalized_x=x)
    (out/"parameters.txt").write_text(
        "// EXPERIMENTAL: requires exact normalized hysteresis update; not old lib3 Euler-z.\n"
        +"// "+("ACCEPTED PILOT" if accepted else "REJECTED: feasibility failed")+"\n"
        +"const ValveModelParams CHAMBER_NEG_PARAMS = {\n    "
        +", ".join(format(v, ".17g") for v in p)+"\n};\n")
    full = np.exp(-p[8]*np.logaddexp(0, -p[1]*(.3+p[4]*np.array([-1., 0., 1.])-p[2])))
    report = dict(accepted=accepted, dataset=str(raw), valve=2, replay_path=str(trace_path), baseline=metrics(baseline_q),
                  candidate=metrics(pred), anchors_passed=sum(a["pass"] for a in anchor_rows),
                  anchors_total=len(anchor_rows), minimum_normalized_margin=float(g.min()),
                  full_open_areas_minus1_zero_plus1=full.tolist(), max_abs_h=float(abs(h).max()),
                  max_abs_force=float(abs(force).max()), parameters=p.tolist(),
                  core_evaluations=pilot.calls, seconds=time.perf_counter()-start,
                  local_optimization=local_results,
                  gates=dict(anchor_allowance_fraction_of_reference=.05, full_open_area_min=.95,
                             flow_max_factor_of_measured_p999=1.2, hysteresis_fraction_beta_min=.5,
                             wn_bounds=[args.wn_min, args.wn_max],
                             zeta_bounds=[args.zeta_min, args.zeta_max]),
                  hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (raw, baseline_path, optimizer, source, library, Path(__file__))},
                  limitations=["In-sample single-valve pilot, not a held-out validation.",
                               "Anchors use fitted trajectory with a declared 5% flow allowance.",
                               "Low u=.8 has <=.2s dwell; static closure threshold is not identified.",
                               "All v2 observations are choked; Cp fixed zero is an identifiability convention.",
                               "Flow uses original mean-dt and 20-substep discretization; sensor delay not identified.",
                               "A/Amax full-open .95 is a model prior, not an independently measured area.",
                               "Changed optimizer and parameterization; not a controlled one-factor ablation.",
                               "Exact hysteresis update must also be ported before any lib3/pump/RL deployment."])
    # Serialization check against the selected full-precision output parameters.
    reread = np.load(out/"parameters.npz")["parameters"]
    assert np.array_equal(p, reread)
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for ax, span in zip(axes[:2], [(0, 657), (497, 531)]):
        ax.plot(data["Time"], data["Q"], color="black", label="Real", lw=1.2)
        ax.plot(data["Time"], baseline_q, label="Previous fit", alpha=.65)
        ax.plot(data["Time"], pred, label="Feasibility pilot", alpha=.85)
        ax.set(xlim=span, ylabel="Flow (L/min)")
    u = np.linspace(.8, 1, 301)
    for history in (-1., 0., 1.):
        area_grid = np.exp(-p[8]*np.logaddexp(0, -p[1]*(.6*u-.3+p[4]*history-p[2])))
        axes[2].plot(u, area_grid*p[0]*pilot.capacity, label=f"Static h={history:g}")
    axes[2].scatter([a["ctrl"] for a in anchor_rows if a["kind"] == "hold_tail"],
                    [a["real_mean"] for a in anchor_rows if a["kind"] == "hold_tail"],
                    s=15, color="black", label="Measured hold tails (not static truth)")
    axes[2].set(xlabel="Physical command", ylabel="Flow (L/min)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="best")
    fig.savefig(out/"comparison.png", dpi=150)
    plt.close(fig)
    plot_surface(pd.read_csv(trace_path), p, out/"surface.png")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
