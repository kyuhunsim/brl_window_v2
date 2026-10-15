#!/usr/bin/env python3
"""Validate a generated lib4 displacement-inclusive oracle reference bank."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (ROOT / "pneu_env" / "src", ROOT / "pneu_ref" / "src", ROOT / "pneu_utils" / "src", SCRIPT_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from generate_lib4_feasible_reference import REFERENCE_INDICES, STATE_NAMES, simulate
from preflight_lib4 import build_sim, load_model_cfg
from pneu_ref.trajectory_ref_nd import TrajectoryBankRefND


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # The bank is intentionally stored as float32 for training.  Replaying a
    # 40.96 s nonlinear rollout from those quantized valve commands produces
    # a small accumulated drift, so exact float64 equality is not a valid
    # oracle test.  0.25 kPa is well below the reference-rate/band margins.
    parser.add_argument("--replay-tolerance", type=float, default=0.25)
    args = parser.parse_args()
    with args.reference_profile.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if profile.get("kind") != "trajectory_bank_nd":
        raise ValueError("reference profile must have kind: trajectory_bank_nd")
    kwargs = dict(profile["kwargs"])
    npz_path = resolve(kwargs["trajectory_npz"])
    with np.load(npz_path) as data:
        time = np.asarray(data["time"], dtype=np.float64)
        reference = np.asarray(data["reference"], dtype=np.float64)
        state = np.asarray(data["state"], dtype=np.float64)
        raw_action = np.asarray(data["raw_action"], dtype=np.float64)
    if reference.ndim == 2:
        reference = reference[None, ...]
    if state.ndim == 2:
        state = state[None, ...]
    if raw_action.ndim == 2:
        raw_action = raw_action[None, ...]
    if time.ndim == 1:
        time = np.repeat(time[None, :], reference.shape[0], axis=0)
    expected_shapes = (reference.shape[:2], state.shape[:2], raw_action.shape[:2], time.shape)
    if len({shape for shape in expected_shapes}) != 1 or reference.shape[2] != 3 or state.shape[2] != len(STATE_NAMES) or raw_action.shape[2] != 6:
        raise ValueError(f"invalid bank shapes: reference={reference.shape}, state={state.shape}, raw_action={raw_action.shape}, time={time.shape}")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(raw_action)):
        raise ValueError("reference bank contains non-finite values")
    if np.any(np.diff(time, axis=1) <= 0.0):
        raise ValueError("reference time is not strictly increasing")
    if not np.allclose(reference, state[:, :, REFERENCE_INDICES], atol=2e-5, rtol=0.0):
        raise ValueError("reference channels do not match stored oracle state")

    design = dict(profile["design"])
    filters = dict(design["filters"])
    bands = filters["state_bands"]
    names = ("ch_pos", "ch_neg", "act_pos", "act_neg", "displacement_mm")
    lower = np.array([bands[name][0] for name in names], dtype=np.float64)
    upper = np.array([bands[name][1] for name in names], dtype=np.float64)
    rate_limits = np.asarray(filters["rate_abs_max"], dtype=np.float64)
    reserve_limit = np.asarray(filters["minimum_reserve_kpa"], dtype=np.float64)
    ignore = int(round(float(filters["ignore_initial_sec"]) * float(design["frequency_hz"])))
    block_steps = int(design["block_steps"])
    details: list[dict[str, Any]] = []
    all_checks: list[bool] = []
    for index in range(reference.shape[0]):
        checked = state[index, min(ignore, len(state[index]) - 1):]
        rate = np.gradient(state[index], 1.0 / float(design["frequency_hz"]), axis=0)[min(ignore, len(state[index]) - 1):]
        reserve = np.c_[checked[:, 0] - checked[:, 2], checked[:, 3] - checked[:, 1]]
        boundaries = np.arange(block_steps, len(checked) + min(ignore, len(state[index]) - 1), block_steps)
        state_band_pass = bool(np.all(checked[:, :5] >= lower) and np.all(checked[:, :5] <= upper))
        rate_pass = bool(np.all(np.max(np.abs(rate[:, :5]), axis=0) <= rate_limits))
        reserve_pass = bool(np.all(reserve.min(axis=0) >= reserve_limit))
        boundary_pass = bool(np.all(np.isfinite(state[index, boundaries]))) if len(boundaries) else True
        details.append({
            "trajectory": index,
            "state_min": checked[:, :5].min(axis=0),
            "state_max": checked[:, :5].max(axis=0),
            "rate_abs_max": np.max(np.abs(rate[:, :5]), axis=0),
            "reserve_min": reserve.min(axis=0),
            "max_reference_step_at_block_boundary": (np.max(np.abs(reference[index, boundaries] - reference[index, boundaries - 1]), axis=0) if len(boundaries) else np.zeros(3)),
            "checks": {"state_band": state_band_pass, "rate": rate_pass, "reserve": reserve_pass, "boundary_finite": boundary_pass},
        })
        all_checks.extend((state_band_pass, rate_pass, reserve_pass, boundary_pass))

    cfg, _ = load_model_cfg(str(design["model_cfg"]))
    source = str(design["simulator"]["state_source"])
    sim, _ = build_sim(cfg, source)
    replay_errors = []
    for index in range(reference.shape[0]):
        replay = simulate(sim, raw_action[index])
        replay_errors.append(float(np.max(np.abs(replay - state[index]))))
    replay_pass = bool(max(replay_errors, default=float("inf")) <= args.replay_tolerance)
    all_checks.append(replay_pass)

    # Constructor-level smoke: PneuRefND expects a stable 3D goal and is what
    # PneuEnv4 uses during training/carry.
    ref = TrajectoryBankRefND(npz_path, reference_key=str(kwargs.get("reference_key", "reference")), seed=int(kwargs.get("seed", 0)), sample_mode="coverage", repeat_after_end=False)
    ref_shape_pass = len(ref.get_goal(float(time[0, min(1, time.shape[1] - 1)]))) == 3
    all_checks.append(ref_shape_pass)
    report = {
        "phase": "phase1_lib4_feasible_reference_validation",
        "reference_profile": str(args.reference_profile),
        "trajectory_npz": str(npz_path),
        "reference_shape": list(reference.shape),
        "oracle_replay_max_abs_error": replay_errors,
        "oracle_replay_pass": replay_pass,
        "nd_reference_dimension_pass": ref_shape_pass,
        "trajectories": details,
        "all_checks_pass": bool(all(all_checks)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable(report), handle, sort_keys=False)
    print(yaml.safe_dump(serializable({"output": str(args.output), "all_checks_pass": report["all_checks_pass"]}), sort_keys=False))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
