"""Audit the two Oct-22 valve fits without optimization or hardware access.

Reads the separate pneu_tune workspace; writes only to --output-dir.
Static z=0 surfaces are diagnostics, not measured dynamic flow envelopes.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reconstruct_history(data, p):
    """Mirror sim_core.cpp's hysteresis update and protection limits."""
    current = data["I"]
    z = np.zeros(len(current))
    for k in range(1, len(current)):
        di = abs(current[k] - current[k-1]) * (2*data["State"][k]-1)
        prev = z[k-1]
        dz = p[5]*di - p[6]*abs(di)*prev - p[7]*di*abs(prev)
        value = prev + dz
        z[k] = min(1e6, max(-1e6, value))
    force0 = current + p[3]*data["P_in_abs"] - p[2]
    force = force0 + p[4]*z
    ld = p[8]*np.logaddexp(0, -p[1]*np.clip(force, -500, 500))
    area = np.exp(-np.minimum(ld, 746))
    area[ld > 700] = 0
    return z, force0, force, area


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-root", type=Path, default=Path("/home/rise/test/pneu_tune"))
    ap.add_argument("--output-dir", type=Path, default=Path("../pneu_tune/exp/lib3_valve_feasibility_20261022"))
    args = ap.parse_args()
    root, out = args.tune_root.resolve(), args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/"src"))
    optimizer_path = root/"src/pneu_tune/optimizer.py"
    spec = importlib.util.spec_from_file_location("valve_audit_optimizer", optimizer_path)
    opt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt)
    rows, curves, bins, sources = [], [], [], {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    detail = []

    def source(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    source(optimizer_path)
    source(root/"src/pneu_tune/tuner/sim_core.cpp")
    source(root/"src/pneu_tune/tuner/sim_core.so")
    for kind in ("type1", "type2"):
        result_file = root/f"tune_result/output_261022_{kind}_baseline.txt"
        text = result_file.read_text()
        raw = root/re.search(r"^// data: (.+)$", text, re.M)[1]
        source(result_file)
        source(raw)
        df = pd.read_csv(raw)
        blocks = re.findall(r"const ValveModelParams\s+(\w+)\s*=\s*\{([^}]+)", text)
        for idx, (name, block) in enumerate(blocks):
            p = np.array([float(v) for v in block.split(",") if v.strip()])
            cfg = opt.CONFIGS[idx]
            data = opt.load_and_preprocess(str(raw), cfg, start_time_sec=0,
                                          input_window_size=1, flow_window_size=1)
            core_loss, pred = opt.simulate_physics_model(data, p)
            n = len(pred)
            z, force0, force, area = reconstruct_history(data, p)
            assert np.isfinite(pred).all() and np.isfinite(z).all()
            assert ((area >= 0) & (area <= 1)).all()
            # Check the first two steps independently: this is where the core adds its jump penalty.
            q_static = p[0]*area*data["P_in_abs"]*data["Phi"]
            x1, x2 = data["Q"][0], 0.0
            for k in range(2):
                wn, damping = (p[9], p[10]) if data["State"][k] == 1 else (p[11], p[12])
                for _ in range(20):
                    dx2 = wn*wn*(q_static[k]-x1)-2*damping*wn*x2
                    x1, x2 = x1+data["dt"]/20*x2, x2+data["dt"]/20*dx2
                assert abs(x1-pred[k]) < 1e-9
            pin = float(np.median(data["P_in_abs"]))
            ratios = opt.static_area_ratio_from_ctrl(np.array([.85, .9, .95, 1]), opt.sanitize_params(p), pin)
            shape = np.sum(np.maximum(np.array([0, .05, .2, .9])-ratios, 0)**2
                           + np.maximum(ratios-np.array([.25, .6, .92, 1]), 0)**2)
            shape += np.sum(np.maximum(.03-np.diff(ratios), 0)**2)
            full_penalty = n*max(0, .95-ratios[-1])**2
            shape_penalty = 5*n*shape if idx in (2, 3) else 0
            sse = float(np.sum((pred-data["Q"])**2))
            input_jump = abs(np.diff(data["I"]))
            stability = (p[6]+p[7])*input_jump
            boundary = (sum(np.isclose(p[[9, 11]], 11) | np.isclose(p[[9, 11]], 25))
                        + sum(np.isclose(p[[10, 12]], .4) | np.isclose(p[[10, 12]], .6)))
            row = dict(kind=kind, valve=idx+1, name=name, rmse_lpm=np.sqrt(sse/n),
                       r2=opt.compute_r2(data["Q"], pred), hf=opt.high_freq_ratio(data["Q"], pred),
                       sse=sse, core_loss=core_loss, core_jump_penalty=core_loss-sse,
                       first_jump_margin=pred[1]-2*np.mean(data["Q"]),
                       full_open_ratio=ratios[-1], full_open_penalty=full_penalty,
                       shape_penalty=shape_penalty,
                       full_penalty_fraction_of_sse=full_penalty/sse,
                       z_limit_pct=100*np.mean(abs(z) >= 1e6),
                       force_limit_pct=100*np.mean(abs(force) > 500),
                       area_extreme_pct=100*np.mean((area < .001) | (area > .999)),
                       hysteresis_term_abs_p95=np.quantile(abs(p[4]*z), .95),
                       other_force_abs_p95=np.quantile(abs(force0), .95),
                       max_euler_branch_factor=float(np.max(stability)),
                       branch_factor_above2_count=int(np.sum(stability > 2)),
                       dynamic_bounds_hit=int(boundary),
                       command_echo_mismatch_pct=100*np.mean(abs(df[f"ctrl{idx+1}"]-df[f"command_ctrl{idx+1}"]) > .001))
            rows.append(row)
            ugrid = np.linspace(.8, 1, 81)
            agrid = opt.static_area_ratio_from_ctrl(ugrid, opt.sanitize_params(p), pin)
            for u, a in zip(ugrid, agrid):
                curves.append(dict(kind=kind, valve=idx+1, pin_kpa=pin, ctrl=u, area_ratio=a))
            ax = axes.flat[idx]
            ax.plot(ugrid, agrid, label=kind)
            ax.set(title=f"v{idx+1}: {name}", xlabel="Physical command", ylabel="Static area / Amax (z=0)", ylim=(-.03, 1.03))
            ax.grid(alpha=.25)
            ax.axhline(.95, color="gray", ls=":", lw=1)
            ax.legend()
            # Descriptive command bins only: pressure and history vary within each bin.
            u = .5+.5*data["u"]
            for center in (.8, .85, .9, .925, .95, .975, 1):
                mask = abs(u-center) < .004
                if mask.any():
                    bins.append(dict(kind=kind, valve=idx+1, ctrl=center, n=int(sum(mask)),
                                     q_real_median=float(np.median(data["Q"][mask])),
                                     q_pred_median=float(np.median(pred[mask])),
                                     pin_median=float(np.median(data["P_in_abs"][mask])),
                                     pout_median=float(np.median(data["P_out_abs"][mask])),
                                     area_history_median=float(np.median(area[mask]))))
            if kind == "type2" and idx == 4:
                # Compact snapshot for the most severe full-open static diagnostic.
                detail.append((data, pred, z, force, area, p))
    fig.savefig(out/"static_opening.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(out/"static_opening.csv", index=False)
    pd.DataFrame(bins).to_csv(out/"command_bins.csv", index=False)
    data, pred, z, force, area, p = detail[0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(data["Time"], data["Q"], label="Real flow", color="black")
    axes[0].plot(data["Time"], pred, label="Fit", alpha=.75)
    axes[0].set(ylabel="Flow (L/min)")
    axes[1].plot(data["Time"], .5+.5*data["u"], label="Command")
    axes[1].plot(data["Time"], area, label="Area ratio with fitted history")
    axes[1].set(ylabel="Command / area ratio")
    axes[2].plot(data["Time"], force, label="Pre-clipped force proxy")
    axes[2].axhline(500, color="red", ls=":", label="Numerical limit ±500")
    axes[2].axhline(-500, color="red", ls=":")
    axes[2].set(xlabel="Recorded time (s)", ylabel="Model force proxy")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=.25)
    fig.savefig(out/"type2_v5_history.png", dpi=160)
    plt.close(fig)
    (out/"audit.json").write_text(json.dumps(dict(sources=sources, limitations=[
        "No parameter optimization or hardware execution performed.",
        "Parameters read from rounded text; discontinuous jump penalty can change after serialization.",
        "Static z=0 area is not a physical ground truth or the full dynamic flow response.",
        "3D actuator surfaces fix the second pressure at its median and hide hysteresis.",
        "Command bins do not control pressure, elapsed settling time, or history.",
        "Euler branch factor is a local same-sign numerical stability diagnostic, not a universal physical constraint.",
    ]), indent=2)+"\n")
    print(pd.DataFrame(rows)[["kind", "valve", "rmse_lpm", "r2", "full_open_ratio", "force_limit_pct", "z_limit_pct", "dynamic_bounds_hit"]].to_string(index=False))


if __name__ == "__main__":
    main()
