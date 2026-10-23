#!/usr/bin/env python3
"""Tune pump coefficients with a type-specific, externally built lib3 library."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneu_env.pump_replay import (CHANNEL_NAMES, INITIAL_GUESS, OBJECTIVE_WEIGHTS,
                                  NativeLib3, PumpReplayTuner, channel_metrics,
                                  load_trajectory, replay)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, action="append", required=True,
                        help="CSV path; repeat to sum objectives across files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--duration-sec", "--duration", type=float)
    parser.add_argument("--frequency", type=float, default=50.)
    parser.add_argument("--initial-inlet", type=float, default=float(INITIAL_GUESS[0]))
    parser.add_argument("--initial-outlet", type=float, default=float(INITIAL_GUESS[1]))
    parser.add_argument("--initial-coeff", type=float, nargs=2, metavar=("INLET", "OUTLET"),
                        help="initial coefficients in micro units; overrides individual initial flags")
    parser.add_argument("--simplex-relative-step", type=float, default=.1,
                        help="positive relative displacement of each initial simplex axis")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--maxfev", type=int)
    parser.add_argument("--xatol", type=float, default=1e-2)
    parser.add_argument("--fatol", type=float, default=1e-2)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.maxiter < 1 or (args.maxfev is not None and args.maxfev < 1):
        parser.error("maxiter/maxfev must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        parser.error(f"output directory already contains a run: {output}")
    datasets = [load_trajectory(path, start_sec=args.start_sec, end_sec=args.end_sec,
                               duration_sec=args.duration_sec) for path in args.data_path]
    library = args.library_path.resolve(strict=True)
    initial = np.array(args.initial_coeff if args.initial_coeff is not None else
                       [args.initial_inlet, args.initial_outlet])
    if not np.isfinite(initial).all() or np.any(initial <= 0):
        parser.error("initial pump coefficients must be finite and positive")
    if not np.isfinite(args.simplex_relative_step) or args.simplex_relative_step <= 0:
        parser.error("simplex-relative-step must be finite and positive")
    config = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": {key: str(value) if isinstance(value, Path) else
                            [str(v) for v in value] if key == "data_path" else value
                            for key, value in vars(args).items()},
              "library_path": str(library), "library_sha256": sha256(library),
              "data": [{"path": str(d.path), "sha256": sha256(d.path),
                        "source_start_sec": float(d.source_time[0]),
                        "source_end_sec": float(d.source_time[-1]),
                        "rows": len(d.time),
                        "duplicate_time_rows_dropped": d.duplicate_time_rows_dropped}
                       for d in datasets],
              "source_sha256": {str(p): sha256(p) for p in
                                [Path(__file__).resolve(),
                                 Path(__file__).resolve().parents[1] / "src/pneu_env/pump_replay.py"]},
              "coefficient_units": "coeff_micro * 1e-6 passed to native simulator",
              "objective_weights": OBJECTIVE_WEIGHTS,
              "replay": {"observation_delay_sec": 0., "noise": False,
                         "control_columns": "ctrl1..ctrl6, physical commands, no scaling",
                         "command_hold": "previous sample at physical interval start",
                         "duplicate_timestamps": "last recorded sample kept, no source file rewrite",
                         "measurement_alignment": "linear interpolation, common pressure/flow times",
                         "initialization": "four measured pressures; native hidden-state reset"}}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sim = NativeLib3(library)
    wall_start = time.monotonic()
    with (output / "evaluations.jsonl").open("w") as history:
        def on_evaluation(record: dict) -> None:
            record["elapsed_sec"] = time.monotonic() - wall_start
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            print(f"evaluation={record['evaluation']} coeff={record['coeff_micro']} "
                  f"objective={record['objective']:.8g} elapsed={record['elapsed_sec']:.1f}s",
                  flush=True)
        tuner = PumpReplayTuner(sim, datasets, args.frequency, on_evaluation)
        if args.evaluate_only:
            value = tuner.objective(initial)
            if value >= 1e12:
                raise RuntimeError("evaluation failed; see evaluations.jsonl")
            coeff = initial
            optimizer_result = None
        else:
            simplex = np.vstack([initial, initial * [1 + args.simplex_relative_step, 1.],
                                 initial * [1., 1 + args.simplex_relative_step]])
            options = {"maxiter": args.maxiter, "xatol": args.xatol, "fatol": args.fatol,
                       "initial_simplex": simplex}
            if args.maxfev is not None:
                options["maxfev"] = args.maxfev
            result = tuner.tune(initial, options)
            coeff = np.asarray(result.x, dtype=float).copy()
            if not np.isfinite(result.fun) or result.fun >= 1e12:
                raise RuntimeError("optimizer found no valid replay; see evaluations.jsonl")
            optimizer_result = {"success": bool(result.success), "status": int(result.status),
                                "message": str(result.message), "x": coeff.tolist(),
                                "fun": float(result.fun), "nit": int(result.nit),
                                "nfev": int(result.nfev)}
    # Preserve the optimizer's selected coefficients before verification/plotting.
    np.savetxt(output / "coefficients_micro.txt", coeff[None], fmt="%.17g",
               header="inlet_pump_coeff_micro outlet_pump_coeff_micro")
    report = {"coeff_micro": coeff.tolist(), "coeff_native": (coeff * 1e-6).tolist(),
              "optimizer": optimizer_result, "datasets": [],
              "fit_status": "evaluation_only" if args.evaluate_only else
                            "converged" if optimizer_result["success"] else "incomplete",
              "fit_scope": "in-sample replay; not a held-out validation"}
    for index, dataset in enumerate(datasets, start=1):
        times, prediction, truth = replay(sim, dataset, coeff, args.frequency)
        metrics = channel_metrics(prediction, truth)
        report["datasets"].append({"path": str(dataset.path), "metrics": metrics,
                                    "objective": sum(w * metrics[n]["mae"]
                                                     for n, w in OBJECTIVE_WEIGHTS.items())})
        data = {"time": times, "source_time": times + dataset.source_time[0]}
        for j, name in enumerate(CHANNEL_NAMES):
            data[f"real_{name}"] = truth[:, j]
            data[f"sim_{name}"] = prediction[:, j]
        pd.DataFrame(data).to_csv(output / f"replay_{index}.csv", index=False)
        if not args.skip_plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True)
            for j, (name, axis) in enumerate(zip(CHANNEL_NAMES, axes.flat)):
                axis.plot(times, truth[:, j], label="real", linewidth=.7)
                axis.plot(times, prediction[:, j], label="sim", linewidth=.7)
                unit = "kPa" if j < 4 else "L/min"
                axis.set_title(f"{name} ({unit}), RMSE={metrics[name]['rmse']:.3f}")
                axis.grid(alpha=.3)
                axis.legend()
            axes[-1, 0].set_xlabel("elapsed time (s)")
            axes[-1, 1].set_xlabel("elapsed time (s)")
            fig.suptitle(f"Pump replay: {dataset.path.name}")
            fig.tight_layout()
            fig.savefig(output / f"comparison_{index}.png", dpi=140)
            plt.close(fig)
    report["elapsed_sec"] = time.monotonic() - wall_start
    report["library_hash_unchanged"] = sha256(library) == config["library_sha256"]
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved coefficients and four-pressure/six-flow verification: {output}", flush=True)


if __name__ == "__main__":
    main()
