"""Analyze action headroom behind a screening trajectory.

This is intentionally a diagnostic, not a replacement controller.  It asks
whether the action still had room to move in the direction that would reduce
the observed pressure error:

* POS deficit (reference > measured): lower ``u_pos`` should raise POS.
* POS excess (reference < measured): higher ``u_pos`` should lower POS.
* NEG deficit: higher ``u_neg`` should raise NEG.
* NEG excess: lower ``u_neg`` should lower NEG.

The result separates action saturation from a policy that simply did not use
available action room.  It cannot by itself prove dynamic feasibility because
pressure delay and POS/NEG coupling still require a local rollout test.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def summarize_direction(mask: np.ndarray, margin: np.ndarray) -> dict:
    if not np.any(mask):
        return {"samples": 0, "ratio": 0.0}
    selected = margin[mask]
    return {
        "samples": int(np.sum(mask)),
        "ratio": float(np.mean(mask)),
        "margin_p10": float(np.quantile(selected, 0.10)),
        "margin_median": float(np.quantile(selected, 0.50)),
        "near_bound_ratio_margin_lt_0.10": float(np.mean(selected < 0.10)),
    }


def analyze(screen_dir: Path, transient_sec: float, tolerance: float) -> dict:
    csv_path = screen_dir / "trajectory.csv"
    metrics_path = screen_dir / "metrics.yaml"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {csv_path}")

    time = values(rows, "time")
    pos_error = values(rows, "pos_error")
    neg_error = values(rows, "neg_error")
    ctrl_pos = values(rows, "ctrl_pos")
    ctrl_neg = values(rows, "ctrl_neg")
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = yaml.safe_load(handle) or {}
    low = float(metrics.get("action_scale_low", 0.85))
    high = float(metrics.get("action_scale_high", 1.0))
    span = max(high - low, 1e-9)

    keep = time >= float(transient_sec)
    ep = pos_error[keep]
    en = neg_error[keep]
    up = ctrl_pos[keep]
    un = ctrl_neg[keep]

    # Fraction of the available normalized action range in the helpful
    # direction.  A value near zero means the action is already at the limit.
    pos_deficit_margin = (up - low) / span
    pos_excess_margin = (high - up) / span
    neg_deficit_margin = (high - un) / span
    neg_excess_margin = (un - low) / span

    pos_deficit = ep > tolerance
    pos_excess = ep < -tolerance
    neg_deficit = en > tolerance
    neg_excess = en < -tolerance

    def worst(
        error: np.ndarray,
        source_keep: np.ndarray,
        reference_key: str,
        measured_key: str,
        control: np.ndarray,
    ) -> dict:
        indices = np.flatnonzero(keep & source_keep)
        if len(indices) == 0:
            return {}
        idx = indices[int(np.argmax(np.abs(error[indices])))]
        reference = values(rows, reference_key)
        measured = values(rows, measured_key)
        return {
            "time_sec": float(time[idx]),
            "reference": float(reference[idx]),
            "measured": float(measured[idx]),
            "error": float(error[idx]),
            "control": float(control[idx]),
        }

    result = {
        "screen_dir": str(screen_dir),
        "transient_excluded_until_sec": float(transient_sec),
        "error_tolerance_kpa": float(tolerance),
        "action_range": {"low": low, "high": high},
        "reference_capacity_check": {
            "joint_rate_violation_ratio": metrics.get("joint_ref_rate_violation_ratio"),
            "joint_rate_utilization_max": metrics.get("joint_ref_rate_utilization_max"),
            "note": "This is copied from screen metrics; action headroom does not prove dynamic feasibility.",
        },
        "pos": {
            "deficit_needs_lower_u_pos": summarize_direction(pos_deficit, pos_deficit_margin),
            "excess_needs_higher_u_pos": summarize_direction(pos_excess, pos_excess_margin),
            "worst_deficit": worst(pos_error, pos_error > tolerance, "ref_pos", "sen_pos", ctrl_pos),
            "worst_excess": worst(pos_error, pos_error < -tolerance, "ref_pos", "sen_pos", ctrl_pos),
        },
        "neg": {
            "deficit_needs_higher_u_neg": summarize_direction(neg_deficit, neg_deficit_margin),
            "excess_needs_lower_u_neg": summarize_direction(neg_excess, neg_excess_margin),
            "worst_deficit": worst(neg_error, neg_error > tolerance, "ref_neg", "sen_neg", ctrl_neg),
            "worst_excess": worst(neg_error, neg_error < -tolerance, "ref_neg", "sen_neg", ctrl_neg),
        },
        "joint_error_geometry": {
            "simultaneous_pos_neg_deficit_ratio": float(np.mean(pos_deficit & neg_deficit)),
            "opposite_sign_large_error_ratio": float(
                np.mean((pos_deficit & neg_excess) | (pos_excess & neg_deficit))
            ),
            "interpretation": (
                "Simultaneous deficits indicate a possible joint capacity problem. "
                "Opposite-sign errors require coordinated two-action changes, but "
                "are not by themselves evidence of infeasibility."
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--transient-sec", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()
    result = analyze(args.screen_dir, args.transient_sec, args.tolerance)
    output = args.screen_dir / "control_opportunity.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(f"[ INFO] wrote {output}")
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
