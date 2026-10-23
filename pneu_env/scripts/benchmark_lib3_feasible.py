#!/usr/bin/env python3
"""Compare two lib3 builds using identical synchronous pump-calibration replay.

This checks numerical parity between libraries, not model accuracy against real
data. Physical models, valve coefficients, and pump coefficients must match for
an optimization comparison. The JSON records the library and data hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneu_env.pump_replay import (CHANNEL_NAMES, INITIAL_GUESS, NativeLib3,
                                  load_trajectory, replay)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--duration", "--duration-sec", type=float)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--coeff", type=float, nargs=2, default=INITIAL_GUESS.tolist(),
                        metavar=("INLET", "OUTLET"), help="pump coefficients in micro units")
    parser.add_argument("--frequency", type=float, default=50.)
    parser.add_argument("--volume-pos", type=float, default=.75, help="supply chamber liters")
    parser.add_argument("--volume-neg", type=float, default=.4, help="supply chamber liters")
    parser.add_argument("--pressure-atol", type=float, default=1e-3, help="absolute parity tolerance, kPa")
    parser.add_argument("--flow-atol", type=float, default=1e-5, help="absolute parity tolerance, L/min")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    for name in ("frequency", "volume_pos", "volume_neg", "pressure_atol", "flow_atol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    coeff = np.asarray(args.coeff, dtype=float)
    if not np.isfinite(coeff).all() or np.any(coeff <= 0):
        parser.error("pump coefficients must be finite and positive")
    libraries = {"reference": args.reference_library.resolve(strict=True),
                 "candidate": args.candidate_library.resolve(strict=True)}
    if libraries["reference"] == libraries["candidate"]:
        parser.error("reference and candidate must be different library files")
    trajectory = load_trajectory(args.data_path, duration_sec=args.duration)
    adapters = {name: NativeLib3(path, args.volume_pos, args.volume_neg)
                for name, path in libraries.items()}
    wall_start = time.perf_counter()
    timings: dict[str, list[float]] = {name: [] for name in libraries}
    differences = np.zeros(len(CHANNEL_NAMES))
    repeat_differences = {name: np.zeros(len(CHANNEL_NAMES)) for name in libraries}
    first_outputs: dict[str, np.ndarray] = {}
    total_abs_error = np.zeros(len(CHANNEL_NAMES))
    total_squared_error = np.zeros(len(CHANNEL_NAMES))
    paired_samples = 0
    worst_times = np.zeros(len(CHANNEL_NAMES))
    execution_order = []
    for repetition in range(args.repeat):
        # Alternate order to reduce the effect of temperature/other CPU work.
        order = ("reference", "candidate") if repetition % 2 == 0 else ("candidate", "reference")
        execution_order.append(list(order))
        outputs = {}
        output_times = {}
        for name in order:
            begin = time.perf_counter()
            times, prediction, _ = replay(adapters[name], trajectory, coeff, args.frequency)
            elapsed = time.perf_counter() - begin
            timings[name].append(elapsed)
            outputs[name] = prediction
            output_times[name] = times
            if name not in first_outputs:
                first_outputs[name] = prediction.copy()
            repeat_differences[name] = np.maximum(repeat_differences[name],
                np.max(np.abs(prediction - first_outputs[name]), axis=0))
            print(f"repeat={repetition + 1} library={name} elapsed={elapsed:.6f}s", flush=True)
        if not np.array_equal(output_times["reference"], output_times["candidate"]):
            raise RuntimeError("library replay timestamps differ")
        error = outputs["candidate"] - outputs["reference"]
        absolute = np.abs(error)
        local_max = np.max(absolute, axis=0)
        indices = np.argmax(absolute, axis=0)
        improved = local_max > differences
        worst_times[improved] = output_times["reference"][indices[improved]]
        differences = np.maximum(differences, local_max)
        total_abs_error += np.sum(absolute, axis=0)
        total_squared_error += np.sum(error * error, axis=0)
        paired_samples += len(error)
    tolerances = np.r_[np.full(4, args.pressure_atol), np.full(6, args.flow_atol)]
    channel_comparisons = {}
    for j, name in enumerate(CHANNEL_NAMES):
        channel_comparisons[name] = {
            "unit": "kPa" if j < 4 else "L/min",
            "max_abs_difference": float(differences[j]),
            "mae_difference": float(total_abs_error[j] / paired_samples),
            "rmse_difference": float(np.sqrt(total_squared_error[j] / paired_samples)),
            "max_difference_time_sec": float(worst_times[j]),
            "absolute_tolerance": float(tolerances[j]),
            "passed": bool(differences[j] <= tolerances[j]),
        }
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Numerical parity and wall-time comparison of two lib3 builds; not real-data accuracy",
        "libraries": {name: {"path": str(path), "sha256": sha256(path)}
                      for name, path in libraries.items()},
        "data": {"path": str(trajectory.path), "sha256": sha256(trajectory.path),
                 "source_start_sec": float(trajectory.source_time[0]),
                 "source_end_sec": float(trajectory.source_time[-1]),
                 "elapsed_duration_sec": float(trajectory.time[-1]),
                 "rows": len(trajectory.time),
                 "duplicate_time_rows_dropped": trajectory.duplicate_time_rows_dropped},
        "pump_coeff_micro": coeff.tolist(),
        "supply_chamber_volumes_liters": {"positive": args.volume_pos, "negative": args.volume_neg},
        "frequency_hz": args.frequency,
        "repeat": args.repeat,
        "execution_order": execution_order,
        "timings_sec": timings,
        "median_wall_time_sec": medians,
        "speedup_reference_over_candidate": medians["reference"] / medians["candidate"],
        "timing_note": "Observed wall times on this machine; concurrent work may affect performance",
        "comparison_samples_per_repeat": paired_samples // args.repeat,
        "channel_comparisons": channel_comparisons,
        "repeat_max_abs_differences": {name: dict(zip(CHANNEL_NAMES, values.tolist()))
                                       for name, values in repeat_differences.items()},
        "passed": bool(np.all(differences <= tolerances)),
        "all_outputs_finite": True,  # replay rejects nonfinite native outputs.
        "wall_time_sec": time.perf_counter() - wall_start,
        "source_sha256": {str(path): sha256(path) for path in
                          [Path(__file__).resolve(), Path(__file__).resolve().parents[1] /
                           "src/pneu_env/pump_replay.py"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"passed={result['passed']} median_speedup={result['speedup_reference_over_candidate']:.3f} "
          f"output={args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
