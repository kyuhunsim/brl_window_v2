#!/usr/bin/env python3
"""Validate continuity and physical constraints of one continuous carry reference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def as_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [POS, NEG]")
    return float(value[0]), float(value[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.reference_profile.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    design = dict(profile.get("design", {}))
    if not design.get("continuous_carry"):
        raise ValueError("reference profile is not marked continuous_carry")
    npz_path = Path(profile["kwargs"]["trajectory_npz"])
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    with np.load(npz_path) as data:
        time = np.asarray(data["time"], dtype=np.float64)
        pressure = np.asarray(data["pressure"], dtype=np.float64)
        action = np.asarray(data["raw_action"], dtype=np.float64)
    if pressure.shape != action.shape or pressure.ndim != 2 or pressure.shape[1] != 2:
        raise ValueError("pressure/action must both have [steps, 2] shape")
    frequency = 1.0 / float(np.median(np.diff(time)))
    horizon_steps = int(round(float(design["horizon_512_sec"]) * frequency))
    startup = int(round(float(design.get("rate_filter_ignore_initial_sec", 0.0)) * frequency))
    checked = pressure[min(startup, len(pressure) - 1):]
    rate = np.gradient(pressure, 1.0 / frequency, axis=0)[min(startup, len(pressure) - 1):]
    pos_band, neg_band = design["pressure_filter_kpa"]["pos"], design["pressure_filter_kpa"]["neg"]
    rate_limit = as_pair(design["rate_filter_kpa_s"], "rate_filter_kpa_s")
    boundaries = np.arange(horizon_steps, len(time), horizon_steps)
    report = {
        "reference_profile": str(args.reference_profile),
        "samples": int(len(time)),
        "duration_sec": float(time[-1] - time[0]),
        "frequency_hz": frequency,
        "reference_initial_pressure_kpa": pressure[0].tolist(),
        "expected_training_initial_pressure_kpa": design.get("expected_training_initial_pressure_kpa"),
        "transition_count": {
            "pos": design.get("pos_schedule", {}).get("transition_count"),
            "neg": design.get("neg_schedule", {}).get("transition_count"),
        },
        "pressure_span_after_startup_kpa": np.ptp(checked, axis=0).tolist(),
        "pressure_rate_abs_max_after_startup_kpa_s": np.max(np.abs(rate), axis=0).tolist(),
        "boundary_count": int(len(boundaries)),
        "max_pressure_step_at_512_boundary_kpa": (
            np.max(np.abs(pressure[boundaries] - pressure[boundaries - 1]), axis=0).tolist() if len(boundaries) else [0.0, 0.0]
        ),
        "max_action_step_at_512_boundary_raw": (
            np.max(np.abs(action[boundaries] - action[boundaries - 1]), axis=0).tolist() if len(boundaries) else [0.0, 0.0]
        ),
    }
    report["pressure_band_pass"] = bool(
        np.all(checked[:, 0] >= float(pos_band[0])) and np.all(checked[:, 0] <= float(pos_band[1]))
        and np.all(checked[:, 1] >= float(neg_band[0])) and np.all(checked[:, 1] <= float(neg_band[1]))
    )
    report["rate_limit_pass"] = bool(
        report["pressure_rate_abs_max_after_startup_kpa_s"][0] <= rate_limit[0]
        and report["pressure_rate_abs_max_after_startup_kpa_s"][1] <= rate_limit[1]
    )
    report["continuous_boundary_pass"] = bool(np.all(np.isfinite(pressure)) and np.all(np.isfinite(action)))
    report["all_checks_pass"] = bool(report["pressure_band_pass"] and report["rate_limit_pass"] and report["continuous_boundary_pass"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(f"[ INFO] carry validation: {args.output}")
    print(f"[ INFO] all checks pass: {report['all_checks_pass']}")


if __name__ == "__main__":
    main()
