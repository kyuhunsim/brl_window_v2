#!/usr/bin/env python3
"""Sweep smooth conflict-free lib4 valve postures from the tuned initial state.

This is a reference-design diagnostic, not a controller.  It identifies which
six-valve directions can move contraction toward unvisited stroke regions
without treating model extrapolation as training evidence.
"""

from __future__ import annotations

import argparse
import itertools
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

from generate_lib4_feasible_reference import quintic, simulate
from preflight_lib4 import build_sim, load_model_cfg


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


def actuator_pair(mode: str, strength: float) -> tuple[float, float]:
    if mode == "in":
        return strength, 0.0
    if mode == "out":
        return 0.0, strength
    return 0.0, 0.0


def smooth_posture(initial: np.ndarray, target: np.ndarray, steps: int, transition_steps: int) -> np.ndarray:
    action = np.tile(target, (steps + 1, 1))
    action[0] = initial
    count = min(max(transition_steps, 1), steps)
    progress = np.arange(count + 1, dtype=np.float64) / count
    action[: count + 1] = initial + (target - initial) * quintic(progress)[:, None]
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cfg", default="0625_lib4_Ours")
    parser.add_argument("--state-source", choices=("model", "current_tuned"), default="current_tuned")
    parser.add_argument("--duration-sec", type=float, default=10.24)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--strengths", default="0.6,0.8,1.0")
    parser.add_argument("--max-displacement-rate", type=float, default=20.0)
    parser.add_argument("--ignore-initial-sec", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg, _ = load_model_cfg(args.model_cfg)
    sim, sim_record = build_sim(cfg, args.state_source)
    freq = float(sim.freq)
    steps = int(round(args.duration_sec * freq))
    transition_steps = int(round(args.transition_sec * freq))
    strengths = tuple(float(item) for item in args.strengths.split(","))
    if steps < 2 or transition_steps < 1 or any(not 0.0 < value <= 1.0 for value in strengths):
        parser.error("duration/transition must be positive and strengths must lie in (0, 1]")

    initial = np.array([0.5, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for ch_pos, ch_neg, pos_mode, neg_mode, strength in itertools.product(
        (0.0, 0.5, 1.0),
        (0.0, 0.5, 1.0),
        ("idle", "in", "out"),
        ("idle", "in", "out"),
        strengths,
    ):
        pos_pair = actuator_pair(pos_mode, strength)
        neg_pair = actuator_pair(neg_mode, strength)
        target = np.array([ch_pos, ch_neg, *pos_pair, *neg_pair], dtype=np.float64)
        action = smooth_posture(initial, target, steps, transition_steps)
        state = simulate(sim, action)
        checked_start = min(int(round(args.ignore_initial_sec * freq)), len(state) - 1)
        checked = state[checked_start:]
        rate = np.gradient(state, 1.0 / freq, axis=0)[checked_start:]
        reserve = np.c_[checked[:, 0] - checked[:, 2], checked[:, 3] - checked[:, 1]]
        finite = bool(np.all(np.isfinite(checked)))
        stroke_safe = bool(finite and np.all((checked[:, 4] >= 0.0) & (checked[:, 4] <= 32.0)))
        reserve_safe = bool(finite and np.all(reserve.min(axis=0) >= np.array([3.0, 3.0])))
        pressure_safe = bool(
            finite
            and np.all(checked[:, 0] <= 210.0)
            and np.all(checked[:, 1] >= 1.0)
            and np.all(checked[:, 2] <= 165.0)
            and np.all(checked[:, 3] >= 15.0)
        )
        rate_safe = bool(
            finite
            and np.max(np.abs(rate[:, 2])) <= 55.0
            and np.max(np.abs(rate[:, 3])) <= 45.0
            and np.max(np.abs(rate[:, 4])) <= args.max_displacement_rate
        )
        rows.append(
            {
                "target_raw": target,
                "ch_pos_raw": ch_pos,
                "ch_neg_raw": ch_neg,
                "act_pos_mode": pos_mode,
                "act_neg_mode": neg_mode,
                "strength": strength,
                "contraction_start_mm": state[0, 4],
                "contraction_min_mm": np.min(checked[:, 4]),
                "contraction_max_mm": np.max(checked[:, 4]),
                "contraction_terminal_mm": checked[-1, 4],
                "max_abs_contraction_rate_mm_s": np.max(np.abs(rate[:, 4])),
                "actuator_pressure_min_kpa": np.min(checked[:, 2:4], axis=0),
                "actuator_pressure_max_kpa": np.max(checked[:, 2:4], axis=0),
                "reserve_min_kpa": np.min(reserve, axis=0),
                "checks": {
                    "finite": finite,
                    "stroke_0_32": stroke_safe,
                    "reserve": reserve_safe,
                    "pressure": pressure_safe,
                    "rate": rate_safe,
                },
                "accepted": finite and stroke_safe and reserve_safe and pressure_safe and rate_safe,
            }
        )

    accepted = [row for row in rows if row["accepted"]]
    accepted.sort(key=lambda row: float(row["contraction_max_mm"]), reverse=True)
    report = {
        "kind": "lib4_initial_state_displacement_capacity_sweep",
        "model_cfg": args.model_cfg,
        "simulator": sim_record,
        "duration_sec": args.duration_sec,
        "transition_sec": args.transition_sec,
        "max_displacement_rate_mm_s": args.max_displacement_rate,
        "ignore_initial_sec": args.ignore_initial_sec,
        "intended_stroke_mm": [0.0, 32.0],
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "accepted_max_contraction_mm": max((float(row["contraction_max_mm"]) for row in accepted), default=float("nan")),
        "top_accepted": accepted[:24],
        "all_candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(serializable(report), sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(serializable({key: report[key] for key in ("candidate_count", "accepted_count", "accepted_max_contraction_mm")}), sort_keys=False))


if __name__ == "__main__":
    main()
