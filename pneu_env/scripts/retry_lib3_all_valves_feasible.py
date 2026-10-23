"""Independent feasibility-first fits for both Oct-22 trajectories and valves 1..6.

Experimental exact-hysteresis core only; no updates to production lib3.
Results and raw traces go to the sibling pneu_tune/exp workspace.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.sparse import csr_matrix

from retry_lib3_valve_feasible import build_core


PTR = ctypes.POINTER(ctypes.c_double)
ROOT = Path(__file__).resolve().parents[2]
TUNE_ROOT = ROOT.parent / "pneu_tune"
CONSTANT_NAMES = ["CHAMBER_POS_PARAMS", "CHAMBER_NEG_PARAMS", "ACT_POS_IN_PARAMS",
                  "ACT_POS_OUT_PARAMS", "ACT_NEG_IN_PARAMS", "ACT_NEG_OUT_PARAMS"]


def area_ratio(p, command, pin, h):
    force = .6*np.asarray(command)-.3+p[3]*np.asarray(pin)+p[4]*np.asarray(h)-p[2]
    return np.exp(-p[8]*np.logaddexp(0, -p[1]*force))


def anchors_for(data, reference):
    u, t = .5+.5*data["u"], data["Time"]
    cuts = np.r_[0, np.flatnonzero(abs(np.diff(u)) > 1e-4)+1, len(u)]
    anchors = []
    allowance = .05*reference
    def add(ids, kind):
        q = data["Q"][ids]
        anchors.append(dict(ids=ids, kind=kind, n=len(ids), ctrl=float(np.median(u[ids])),
                            start=float(t[ids[0]]), end=float(t[ids[-1]]),
                            lo=float(np.quantile(q, .05)-allowance),
                            hi=float(np.quantile(q, .95)+allowance),
                            real_mean=float(q.mean()), branch=int(data["State"][ids[-1]]),
                            pin=float(np.median(data["P_in_abs"][ids])),
                            pout=float(np.median(data["P_out_abs"][ids]))))
    for a, b in zip(cuts[:-1], cuts[1:]):
        if t[b-1]-t[a]+data["dt"] >= 1.5:
            ids = np.arange(a, b)
            ids = ids[t[ids] >= t[b-1]-.5]
            if len(ids) >= 10:
                add(ids, "hold_tail")
    changed = np.r_[True, abs(np.diff(u)) > 1e-4]
    last = np.maximum.accumulate(np.where(changed, np.arange(len(u)), 0))
    age = (np.arange(len(u))-last)*data["dt"]
    mask = (u > .999) & (age >= 1)
    # Two-pressure bins: unlike the pilot, not restricted to v2's Pout=20..50.
    keys = np.column_stack((np.floor(data["P_in_abs"]/5), np.floor(data["P_out_abs"]/5))).astype(int)
    for key in np.unique(keys[mask], axis=0):
        ids = np.flatnonzero(mask & np.all(keys == key, axis=1))
        if len(ids) >= 25:
            add(ids, "full_open_pressure_bin")
    return anchors


class Fit:
    def __init__(self, core, data, config, settings):
        self.core, self.data, self.config, self.settings = core, data, config, settings
        u = .5+.5*data["u"]
        capacity = data["P_in_abs"]*data["Phi"]
        self.reference = max(float(np.quantile(np.maximum(data["Q"], 0), .95)), .1)
        full = (u > .999) & (capacity > .02*np.max(capacity)) & (data["Q"] > 0)
        if sum(full) < 25:
            raise ValueError("Insufficient full-open positive-pressure capacity samples")
        self.a_reference = float(np.median(data["Q"][full]/capacity[full]))
        self.pin_ref = float(np.median(data["P_in_abs"]))
        self.pin_span = float(np.ptp(data["P_in_abs"]))
        self.variable_pin = self.pin_span > .01
        self.dimension = 12 if self.variable_pin else 11
        self.pin_grid = np.unique(np.quantile(data["P_in_abs"], [0, .5, 1]))
        self.anchors = anchors_for(data, self.reference)
        if not self.anchors:
            raise ValueError("No measured hold/pressure anchors")
        rr, cc, vv = [], [], []
        for j, anchor in enumerate(self.anchors):
            ids = anchor["ids"]
            rr.extend([j]*len(ids)); cc.extend(ids); vv.extend([1/len(ids)]*len(ids))
        self.matrix = csr_matrix((vv, (rr, cc)), shape=(len(self.anchors), len(u)))
        self.lo = np.array([a["lo"] for a in self.anchors])
        self.hi = np.array([a["hi"] for a in self.anchors])
        self.margin_count = 2*len(self.anchors)+3*len(self.pin_grid)+3
        self.max_q = 1.2*float(np.quantile(data["Q"], .999))
        self.arrays = [np.ascontiguousarray(data[k], dtype=np.float64)
                       for k in ("I", "State", "P_in_abs", "Phi", "Q")]
        self.ptrs = [a.ctypes.data_as(PTR) for a in self.arrays]
        self.buffers = [np.empty(len(u)) for _ in range(4)]
        self.output_ptrs = [a.ctypes.data_as(PTR) for a in self.buffers]
        self.cache = None
        self.calls = 0
        self.best_feasible = None
        self.best_near = None
        self.best_loss = None
        self.invalid_counts = {}

    def parameters(self, x):
        k = 5*np.exp(x[1]*np.log(40))
        alpha = .01*np.exp(x[2]*np.log(20000/.01))
        u50 = .7+.35*x[3]
        effective_ck = .6*u50-.3+np.log(np.expm1(np.log(2)/alpha))/k
        cp = .12/self.pin_span*(2*x[11]-1) if self.variable_pin else 0.
        rate = np.exp(x[5]*np.log(2000))
        fraction = .5+.5*x[6]
        wnlo, wnhi = self.settings["wn_bounds"]
        zlo, zhi = self.settings["zeta_bounds"]
        return np.array([self.a_reference*(.85+.4*x[0]), k, effective_ck+cp*self.pin_ref,
                         cp, .06*(2*x[4]-1), rate, rate*fraction, rate*(1-fraction), alpha,
                         wnlo+(wnhi-wnlo)*x[7], zlo+(zhi-zlo)*x[8],
                         wnlo+(wnhi-wnlo)*x[9], zlo+(zhi-zlo)*x[10]])

    def encode(self, p):
        x = np.full(self.dimension, .5)
        x[0] = (p[0]/self.a_reference-.85)/.4
        x[1] = np.log(max(p[1], 5)/5)/np.log(40)
        x[2] = np.log(max(p[8], .01)/.01)/np.log(20000/.01)
        force50 = -np.log(np.expm1(np.log(2)/max(p[8], .01)))/max(p[1], 5)
        u50 = (p[2]-p[3]*self.pin_ref+force50+.3)/.6
        x[3] = (u50-.7)/.35
        rate = max(p[6]+p[7], 1.)
        amplitude = p[4]*p[5]/rate
        x[4] = (amplitude/.06+1)/2
        x[5] = np.log(rate)/np.log(2000)
        x[6] = 2*(p[6]/rate-.5)
        wnlo, wnhi = self.settings["wn_bounds"]
        zlo, zhi = self.settings["zeta_bounds"]
        x[[7, 9]] = (p[[9, 11]]-wnlo)/(wnhi-wnlo)
        x[[8, 10]] = (p[[10, 12]]-zlo)/(zhi-zlo)
        if self.variable_pin:
            x[11] = (p[3]*self.pin_span/.12+1)/2
        return np.clip(x, 0, 1)

    def evaluate(self, x):
        if self.cache is not None and np.array_equal(x, self.cache):
            return self.loss, self.margins
        self.cache = np.array(x, copy=True)
        self.calls += 1
        p = self.parameters(x)
        if p[2] < 0 or not np.isfinite(p).all():
            status = 1
        else:
            status = self.core.simulate_feasible(len(self.arrays[0]), *self.ptrs,
                self.data["dt"], p.ctypes.data_as(PTR), *self.output_ptrs)
        if status:
            self.invalid_counts[str(status)] = self.invalid_counts.get(str(status), 0)+1
            self.loss, self.margins = 1000.+status, np.full(self.margin_count, -1.)
            return self.loss, self.margins
        q, h, _, force = self.buffers
        means = self.matrix @ q
        full = area_ratio(p, 1., self.pin_grid[:, None], np.array([-1., 0., 1.]))
        self.margins = np.r_[(means-self.lo)/self.reference, (self.hi-means)/self.reference,
                             full.ravel()-.95, (q.min()+.01*self.reference)/self.reference,
                             (self.max_q-q.max())/self.reference, 1+1e-10-abs(h).max()]
        self.loss = float(np.mean((q-self.data["Q"])**2)/self.reference**2)
        violation = np.maximum(-self.margins, 0)
        near_score = (float(violation.max()), float(np.sum(violation**2)), self.loss)
        if self.best_near is None or near_score < self.best_near[0]:
            self.best_near = (near_score, x.copy())
        if self.best_loss is None or self.loss < self.best_loss[0]:
            self.best_loss = (self.loss, x.copy())
        if self.margins.min() >= -1e-8:
            if self.best_feasible is None or self.loss < self.best_feasible[0]:
                self.best_feasible = (self.loss, x.copy())
        return self.loss, self.margins

    def screening(self, x):
        loss, margins = self.evaluate(x)
        return loss+1000*float(np.sum(np.minimum(margins, 0)**2))


def read_parameters(path, valve):
    block = re.search(CONSTANT_NAMES[valve-1]+r"\s*=\s*\{([^}]+)", path.read_text())[1]
    return np.array([float(v) for v in block.split(",") if v.strip()])


def curve_metrics(q, real):
    e = q-real
    variance = np.sum((real-real.mean())**2)
    return dict(rmse_lpm=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/variance) if variance else None,
                hf_ratio=float(np.std(np.diff(q))/max(np.std(np.diff(real)), 1e-12)),
                q_min=float(q.min()), q_max=float(q.max()))


def render(data, q, baseline, p, anchors, out, label):
    plt.rcdefaults()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    partial = [a for a in anchors if a["kind"] == "hold_tail" and a["ctrl"] < .98]
    window = (partial[0]["start"]-2, partial[min(7, len(partial)-1)]["end"]+2) if partial else (0, 40)
    for ax, span in zip(axes[:2], [(float(data["Time"][0]), float(data["Time"][-1])), window]):
        ax.plot(data["Time"], data["Q"], color="black", label="Real", lw=1)
        ax.plot(data["Time"], baseline, label="Previous fit", alpha=.65)
        ax.plot(data["Time"], q, label="Feasibility candidate", alpha=.8)
        ax.set(xlim=span, ylabel="Flow (L/min)")
    u = np.linspace(.8, 1, 150)
    pin = float(np.median(data["P_in_abs"]))
    pout = float(np.median(data["P_out_abs"]))
    ratio = min(max(pout/pin, 0), 1)
    phi = np.sqrt(1.4*(2/2.4)**6) if ratio <= (2/2.4)**3.5 else np.sqrt(max(0, 7*(ratio**(2/1.4)-ratio**(2.4/1.4))))
    for h in (-1., 0., 1.):
        axes[2].plot(u, p[0]*area_ratio(p, u, pin, h)*pin*phi, label=f"h={h:g}")
    axes[2].set(xlabel="Physical command", ylabel="Static flow (L/min)",
                title=f"Slice at Pin={pin:.1f}, Pout={pout:.1f} kPa (not all measured conditions)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.suptitle(label)
    fig.savefig(out/"comparison.png", dpi=140); plt.close(fig)
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    variable_in = np.ptp(data["P_in_abs"]) >= np.ptp(data["P_out_abs"])
    pressure = data["P_in_abs"] if variable_in else data["P_out_abs"]
    pg, ug = np.meshgrid(np.linspace(float(pressure.min()), float(pressure.max()), 35), u)
    pi = pg if variable_in else np.full_like(pg, pin)
    po = np.full_like(pg, pout) if variable_in else pg
    r = np.clip(po/pi, 0, 1)
    ph = np.where(r <= (2/2.4)**3.5, np.sqrt(1.4*(2/2.4)**6),
                  np.sqrt(np.maximum(0, 7*(r**(2/1.4)-r**(2.4/1.4)))))
    ax.plot_surface(pg, ug, p[0]*area_ratio(p, ug, pi, 0)*pi*ph, alpha=.2, color="steelblue")
    ids = np.arange(0, len(q), max(1, len(q)//2000))
    raw_u = .5+.5*data["u"]
    ax.scatter(pressure[ids], raw_u[ids], data["Q"][ids], s=3, color="black", label="Real")
    ax.scatter(pressure[ids], raw_u[ids], q[ids], s=3, color="tomato", label="Dynamic fit")
    ax.set(xlabel="Pin (kPa)" if variable_in else "Pout (kPa)", ylabel="Command", zlabel="Flow (L/min)",
           title=label+"; h=0 surface, other pressure fixed at median")
    ax.legend(); fig.savefig(out/"surface.png", dpi=140); plt.close(fig)


def run_one(job):
    kind, valve, settings = job
    started = time.perf_counter()
    root = Path(settings["tune_root"])
    out = Path(settings["output_dir"])/kind/f"v{valve}"
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(settings["trace_root"])/kind/f"v{valve}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (out/"progress.log").open("w") as log:
        def progress(message):
            print(message, file=log, flush=True)
        sys.path.insert(0, str(root/"src"))
        optimizer = root/"src/pneu_tune/optimizer.py"
        spec = importlib.util.spec_from_file_location("original_optimizer", optimizer)
        opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)
        raw = next((root/"exp").glob(f"261022*{kind}*.csv"))
        data = opt.load_and_preprocess(str(raw), opt.CONFIGS[valve-1], start_time_sec=0,
                                      input_window_size=1, flow_window_size=1)
        baseline_path = root/f"tune_result/output_261022_{kind}_baseline.txt"
        baseline_p = read_parameters(baseline_path, valve)
        _, baseline = opt.simulate_physics_model(data, baseline_p)
        core = ctypes.CDLL(settings["library"])
        core.simulate_feasible.argtypes = [ctypes.c_int]+[PTR]*5+[ctypes.c_double]+[PTR]*5
        core.simulate_feasible.restype = ctypes.c_int
        fit = Fit(core, data, opt.CONFIGS[valve-1], settings)
        rng = np.random.default_rng(0)
        size = 8*fit.dimension
        population = rng.uniform(0, 1, (size, fit.dimension))
        seed = fit.encode(baseline_p)
        population[0] = seed
        population[1:15] = np.clip(seed+rng.normal(0, .12, (14, fit.dimension)), 0, 1)
        if valve == 2 and kind == "type1":
            pilot = root / "exp/lib3_v2_feasible_retry_20261022_slow/parameters.npz"
            if pilot.exists():
                population[15] = fit.encode(np.load(pilot)["parameters"])
        # Feasible-ish full-open, small-history seeds avoid only searching legacy saturation basins.
        for j, mid in enumerate((.84, .87, .90, .93), start=16):
            population[j] = .5
            population[j, 1] = np.log(100/5)/np.log(40)
            population[j, 2] = np.log(.1/.01)/np.log(20000/.01)
            population[j, 3] = (mid-.7)/.35
            if fit.variable_pin:
                population[j, 11] = .5
        generation = 0
        def callback(x, convergence):
            nonlocal generation
            generation += 1
            if generation % 20 == 0:
                loss, g = fit.evaluate(x)
                progress(f"DE {generation}: rmse={np.sqrt(loss)*fit.reference:.4f}, margin={g.min():.6g}, feasible={fit.best_feasible is not None}")
        de = differential_evolution(fit.screening, [(0, 1)]*fit.dimension,
            init=population, maxiter=settings["de_iterations"], seed=0, polish=False,
            callback=callback, tol=1e-6)
        seeds = [de.x]
        for candidate in (fit.best_near, fit.best_loss, fit.best_feasible):
            if candidate is not None and all(not np.allclose(candidate[1], s) for s in seeds):
                seeds.append(candidate[1].copy())
        local = []
        for index, initial in enumerate(seeds[:3]):
            result = minimize(lambda x: fit.evaluate(x)[0], initial, method="SLSQP",
                bounds=[(0, 1)]*fit.dimension,
                constraints=[dict(type="ineq", fun=lambda x: fit.evaluate(x)[1])],
                options=dict(maxiter=settings["local_iterations"], ftol=1e-10))
            loss, margins = fit.evaluate(result.x)
            item = dict(success=bool(result.success), message=str(result.message),
                        iterations=int(result.nit), rmse=float(np.sqrt(loss)*fit.reference),
                        min_margin=float(margins.min()))
            local.append(item); progress(f"SLSQP {index+1}: {item}")
        if fit.best_near is None:
            raise RuntimeError("No numerically valid candidate found")
        accepted = fit.best_feasible is not None
        chosen = fit.best_feasible[1] if accepted else fit.best_near[1]
        fit.cache = None
        loss, margins = fit.evaluate(chosen)
        p = fit.parameters(chosen)
        q, h, area, force = [a.copy() for a in fit.buffers]
        means = fit.matrix @ q
        baseline_means = fit.matrix @ baseline
        rows = []
        for j, anchor in enumerate(fit.anchors):
            row = {k: v for k, v in anchor.items() if k != "ids"}
            row.update(pred_mean=float(means[j]), baseline_mean=float(baseline_means[j]),
                       passed=bool(fit.lo[j]-1e-8*fit.reference <= means[j] <= fit.hi[j]+1e-8*fit.reference))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/"anchors.csv", index=False)
        full = area_ratio(p, 1., fit.pin_grid[:, None], np.array([-1., 0., 1.]))
        pd.DataFrame(full, index=fit.pin_grid, columns=["h_minus1", "h_zero", "h_plus1"]).to_csv(out/"full_open_grid.csv", index_label="pin")
        np.savez(out/"parameters.npz", parameters=p, normalized_x=chosen)
        (out/"parameters.txt").write_text("// EXPERIMENTAL EXACT NORMALIZED HYSTERESIS; "+
            ("ACCEPTED FIT" if accepted else "REJECTED: constraints failed")+
            "\nconst ValveModelParams "+CONSTANT_NAMES[valve-1]+" = {\n    "+
            ", ".join(format(v, ".17g") for v in p)+"\n};\n")
        trace_path = trace_dir/"replay.csv"
        pd.DataFrame(dict(time=data["Time"], real=data["Q"], candidate=q, baseline=baseline,
            ctrl=.5+.5*data["u"], pin=data["P_in_abs"], pout=data["P_out_abs"],
            h=h, area=area, force=force)).to_csv(trace_path, index=False)
        baseline_metrics, candidate_metrics = curve_metrics(baseline, data["Q"]), curve_metrics(q, data["Q"])
        report = dict(type=kind, valve=valve, accepted=accepted, baseline=baseline_metrics,
            candidate=candidate_metrics, anchors_passed=sum(a["passed"] for a in rows),
            anchors_total=len(rows), full_open_grid_min=float(full.min()),
            full_open_grid_pass=bool(full.min() >= .95-1e-8), max_abs_h=float(abs(h).max()),
            max_abs_force=float(abs(force).max()), numerical_valid=True, clipping_used=False,
            minimum_normalized_margin=float(margins.min()), selection="best feasible SSE" if accepted else "least maximum constraint violation",
            pressure_pin_range=[float(data["P_in_abs"].min()), float(data["P_in_abs"].max())],
            pressure_pout_range=[float(data["P_out_abs"].min()), float(data["P_out_abs"].max())],
            variable_pin=fit.variable_pin, parameters=p.tolist(), core_evaluations=fit.calls,
            invalid_candidate_counts=fit.invalid_counts, seconds=time.perf_counter()-started,
            local_optimization=local, replay_path=str(trace_path), reference_lpm=fit.reference,
            a_reference=fit.a_reference, settings=settings,
            hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (raw, baseline_path, optimizer, Path(settings["source"]), Path(__file__))},
            limitations=["In-sample fit only; no real controller or held-out validation.",
                "95% area is a prior; empirical anchors constrain recorded flow, not measured area.",
                "Full trace and anchor quantiles retain small negative measured sensor values.",
                "Wn=2..25,zeta=.4..1.2; bounds are exploratory, not hardware-identified.",
                "Missing static closure evidence is not replaced with zero-flow assumptions.",
                "Current/sensor timing and legacy mean-dt one-step output convention are unchanged.",
                "Rejected ranking favors constraint violation, so rejected RMSE need not improve.",
                "Exact update must be ported and replay checked before lib3/pump/RL use."])
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        render(data, q, baseline, p, rows, out, f"{kind} v{valve}: "+("PASS" if accepted else "REJECTED"))
        summary = dict(type=kind, valve=valve, accepted=accepted,
            old_rmse=baseline_metrics["rmse_lpm"], rmse=candidate_metrics["rmse_lpm"],
            old_r2=baseline_metrics["r2"], r2=candidate_metrics["r2"],
            hf=candidate_metrics["hf_ratio"], anchors_passed=report["anchors_passed"],
            anchors_total=len(rows), full_open_min=report["full_open_grid_min"],
            min_margin=report["minimum_normalized_margin"], seconds=report["seconds"],
            report=str(out/"report.json"))
        progress(str(summary))
        return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", choices=["type1", "type2", "all"], default="all")
    ap.add_argument("--valves", default="1,2,3,4,5,6")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--de-iterations", type=int, default=100)
    ap.add_argument("--local-iterations", type=int, default=250)
    ap.add_argument("--output-dir", type=Path, default=TUNE_ROOT / "exp/lib3_all_valves_feasible_20261022")
    ap.add_argument("--trace-root", type=Path, default=TUNE_ROOT / "exp/lib3_all_valves_feasible_20261022/traces")
    ap.add_argument("--tune-root", type=Path, default=TUNE_ROOT)
    args = ap.parse_args()
    valves = [int(v) for v in args.valves.split(",")]
    if not valves or any(v not in range(1, 7) for v in valves):
        ap.error("valves must be 1..6")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, source, library = build_core()
    settings = dict(tune_root=str(args.tune_root.resolve()), output_dir=str(args.output_dir),
                    trace_root=str(args.trace_root), library=str(library), source=str(source),
                    de_iterations=args.de_iterations, local_iterations=args.local_iterations,
                    wn_bounds=[2., 25.], zeta_bounds=[.4, 1.2], seed=0)
    kinds = ["type1", "type2"] if args.type == "all" else [args.type]
    started = time.perf_counter()
    summaries, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (kind, v, settings)): (kind, v)
                   for kind in kinds for v in valves}
        for future in as_completed(futures):
            kind, valve = futures[future]
            try:
                summary = future.result(); summaries.append(summary)
                pd.DataFrame(summaries).sort_values(["type", "valve"]).to_csv(args.output_dir/"summary.csv", index=False)
                print(f"DONE {kind} v{valve}: accepted={summary['accepted']} RMSE={summary['old_rmse']:.3f}->{summary['rmse']:.3f} R2={summary['r2']:.3f} anchors={summary['anchors_passed']}/{summary['anchors_total']}", flush=True)
            except Exception as exc:
                import traceback
                errors.append(dict(type=kind, valve=valve, error=repr(exc)))
                print(traceback.format_exc(), flush=True)
    final = dict(seconds=time.perf_counter()-started, completed=len(summaries),
                 requested=len(kinds)*len(valves), accepted=sum(s["accepted"] for s in summaries),
                 errors=errors, settings=settings)
    (args.output_dir/"batch.json").write_text(json.dumps(final, indent=2)+"\n")
    print(json.dumps(final, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
