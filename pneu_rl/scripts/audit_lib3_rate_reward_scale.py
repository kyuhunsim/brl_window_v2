#!/usr/bin/env python3
"""Quantify how a candidate lib3 rate reward compares with tracking reward.

The analysis is deliberately post-hoc: it does not change policy observation
or reward.  It uses saved evaluation rollouts to choose a rate coefficient
before starting a reward ablation.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def summary_for_files(
    files: list[Path], coeffs: list[float], chamber_bound_coeff: float
) -> dict[str, object]:
    tracking_all: list[np.ndarray] = []
    chamber_bound_all: list[np.ndarray] = []
    rate_loss_all: list[np.ndarray] = []
    plateau_all: list[np.ndarray] = []

    for path in files:
        data = pd.read_csv(path)
        required = {
            "time",
            "ref_pos",
            "ref_neg",
            "act_pos",
            "act_neg",
            "chamber_pos",
            "chamber_neg",
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        time = data["time"].to_numpy(dtype=np.float64)
        if len(time) < 4:
            raise ValueError(f"{path} has fewer than four samples")
        dt = float(np.median(np.diff(time)))
        ref_pos = data["ref_pos"].to_numpy(dtype=np.float64)
        ref_neg = data["ref_neg"].to_numpy(dtype=np.float64)
        act_pos = data["act_pos"].to_numpy(dtype=np.float64)
        act_neg = data["act_neg"].to_numpy(dtype=np.float64)

        tracking = 0.3 * (np.abs(ref_pos - act_pos) + np.abs(ref_neg - act_neg))
        chamber_bound = chamber_bound_coeff * (
            np.maximum(101.325 - data["chamber_pos"].to_numpy(dtype=np.float64), 0.0)
            + np.maximum(data["chamber_pos"].to_numpy(dtype=np.float64) - 220.0, 0.0)
            + np.maximum(10.0 - data["chamber_neg"].to_numpy(dtype=np.float64), 0.0)
            + np.maximum(data["chamber_neg"].to_numpy(dtype=np.float64) - 101.325, 0.0)
        )
        rate_loss = (
            np.abs(np.gradient(act_pos, dt) - np.gradient(ref_pos, dt)) / 12.0
            + np.abs(np.gradient(act_neg, dt) - np.gradient(ref_neg, dt)) / 6.0
        )
        stationary = (
            (np.abs(np.gradient(ref_pos, dt)) < 1.0)
            & (np.abs(np.gradient(ref_neg, dt)) < 0.5)
        )
        stationary[: min(int(round(1.0 / dt)), len(stationary))] = False

        tracking_all.append(tracking)
        chamber_bound_all.append(chamber_bound)
        rate_loss_all.append(rate_loss)
        plateau_all.append(stationary)

    tracking = np.concatenate(tracking_all)
    chamber_bound = np.concatenate(chamber_bound_all)
    rate_loss = np.concatenate(rate_loss_all)
    plateau = np.concatenate(plateau_all)

    def region(mask: np.ndarray) -> dict[str, float]:
        track = tracking[mask]
        bound = chamber_bound[mask]
        rate = rate_loss[mask]
        mean_track = float(np.mean(track))
        mean_base = float(np.mean(track + bound))
        mean_rate = float(np.mean(rate))
        return {
            "samples": int(mask.sum()),
            "tracking_penalty_mean": mean_track,
            "tracking_penalty_median": float(np.median(track)),
            "chamber_bound_penalty_mean": float(np.mean(bound)),
            "base_penalty_mean": mean_base,
            "rate_loss_mean_before_coeff": mean_rate,
            "rate_loss_median_before_coeff": float(np.median(rate)),
            "rate_loss_p95_before_coeff": float(np.quantile(rate, 0.95)),
            "actuator_rate_error_rms_normalized": rms(rate),
            "lambda_for_rate_to_be_10pct_of_tracking": (
                0.10 * mean_base / mean_rate if mean_rate > 1e-12 else float("nan")
            ),
            "lambda_for_rate_to_be_25pct_of_tracking": (
                0.25 * mean_base / mean_rate if mean_rate > 1e-12 else float("nan")
            ),
        }

    result: dict[str, object] = {
        "files": [str(path) for path in files],
        "formula": {
            "tracking_penalty": "0.3 * (abs(ref_pos-act_pos) + abs(ref_neg-act_neg))",
            "chamber_bound_penalty": (
                f"{chamber_bound_coeff:g} * sum(kPa outside configured chamber envelope)"
            ),
            "rate_loss_before_coeff": "abs(d_act_pos-d_ref_pos)/12 + abs(d_act_neg-d_ref_neg)/6",
            "plateau": "after first 1 s, abs(d_ref_pos)<1 kPa/s and abs(d_ref_neg)<0.5 kPa/s",
        },
        "all_steps": region(np.ones_like(plateau, dtype=bool)),
        "plateau": region(plateau),
        "ramp_or_transition": region(~plateau),
        "candidate_coefficients": {},
    }
    candidate: dict[str, object] = {}
    for coefficient in coeffs:
        penalty = coefficient * rate_loss
        candidate[f"{coefficient:g}"] = {
            "rate_penalty_mean": float(np.mean(penalty)),
            "rate_to_base_mean_ratio": float(
                np.mean(penalty) / max(np.mean(tracking + chamber_bound), 1e-12)
            ),
            "plateau_rate_penalty_mean": float(np.mean(penalty[plateau])),
            "plateau_rate_to_base_mean_ratio": float(
                np.mean(penalty[plateau])
                / max(np.mean((tracking + chamber_bound)[plateau]), 1e-12)
            ),
            "rate_penalty_p95": float(np.quantile(penalty, 0.95)),
        }
    result["candidate_coefficients"] = candidate
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        action="append",
        required=True,
        help="glob of rollout CSV files; repeat to combine models or banks",
    )
    parser.add_argument(
        "--coeff",
        type=float,
        action="append",
        default=[0.01, 0.02, 0.05, 0.10],
        help="candidate rate coefficient; repeatable",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--chamber-bound-coeff",
        type=float,
        default=0.10,
        help="common chamber-bound coefficient in the proposed base objective",
    )
    args = parser.parse_args()

    files = sorted({Path(path) for pattern in args.glob for path in glob.glob(pattern)})
    if not files:
        parser.error("no rollout CSV files matched --glob")
    result = summary_for_files(files, args.coeff, args.chamber_bound_coeff)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
