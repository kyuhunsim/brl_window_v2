#!/usr/bin/env python3
"""Validate and visualize every member of a simulator-backed reference bank.

The generator writes a full pressure/action trajectory, but its large YAML is
not convenient for a design review.  This command produces a compact, auditable
report for all bank members: POS and NEG are plotted on separate axes, every
pressure/rate constraint is recomputed after the declared startup window, and
oracle-replay results are joined when available.

It validates the *reference distribution*, not an RL policy.  A pass means the
stored trajectory obeys the configured design filters and its stored action
replays in the simulator; it does not claim that a fixed controller will track
it well.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "pneu_rl/config/reference/lib2_optimized_target_bank.yaml"
DEFAULT_ORACLE = ROOT / "pneu_rl/exp/_analysis/lib2_optimized_target_reference/oracle_replay_target_start/oracle_replay_metrics.csv"
DEFAULT_OUTPUT = ROOT / "pneu_rl/exp/_analysis/lib2_optimized_target_reference/reference_bank_validation"


def _as_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [POS, NEG]")
    return float(value[0]), float(value[1])


def _wrapped_phase_difference(first: float, second: float) -> float:
    if not math.isfinite(first) or not math.isfinite(second):
        return float("nan")
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _schedule_timing(meta: dict[str, Any], axis: str) -> dict[str, Any]:
    schedule = dict(meta.get(f"{axis}_schedule", {}))
    transitions = [float(value) for value in schedule.get("transition_duration_sec", [])]
    holds = [float(value) for value in schedule.get("hold_duration_sec", [])]
    segments = list(schedule.get("segments", []))
    boundaries = [float(item.get("start_sec", float("nan"))) for item in segments]
    return {
        "count": int(schedule.get("realized_transition_count", len(transitions))),
        "transition_values": transitions,
        "hold_values": holds,
        "boundaries": boundaries,
        "transition_mean": float(np.mean(transitions)) if transitions else float("nan"),
        "hold_mean": float(np.mean(holds)) if holds else float("nan"),
    }


def _same_timing(first: list[float], second: list[float]) -> bool:
    return len(first) == len(second) and bool(np.allclose(first, second, rtol=0.0, atol=1e-6))


def _load_oracle(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["trajectory_id"]): {
                "pos_rmse": float(row["pos_rmse"]),
                "neg_rmse": float(row["neg_rmse"]),
                "pos_max_abs_error": float(row["pos_max_abs_error"]),
                "neg_max_abs_error": float(row["neg_max_abs_error"]),
            }
            for row in rows
        }


def _load_profile(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if profile.get("kind") != "optimized_trajectory_bank":
        raise ValueError(f"expected optimized_trajectory_bank, got {profile.get('kind')!r}")
    trajectories = list(profile.get("kwargs", {}).get("trajectories", []))
    if not trajectories:
        raise ValueError("profile contains no inline trajectories")
    return profile, trajectories


def _validate(
    profile: dict[str, Any],
    trajectories: list[dict[str, Any]],
    oracle: dict[int, dict[str, float]],
    profile_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    design = dict(profile.get("design", {}))
    frequency = float(design["frequency_hz"])
    dt = 1.0 / frequency
    start = int(round(float(design.get("rate_filter_ignore_initial_sec", 0.0)) * frequency))
    pos_band, neg_band = _as_pair(design["pressure_filter_kpa"]["pos"], "POS pressure filter") , _as_pair(design["pressure_filter_kpa"]["neg"], "NEG pressure filter")
    rate_pos_max, rate_neg_max = _as_pair(design["rate_filter_kpa_s"], "rate filter")
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(trajectories):
        trajectory_id = int(item.get("id", index))
        time = np.asarray(item["time"], dtype=np.float64)
        pressure = np.asarray(item["pressure"], dtype=np.float64)
        action = np.asarray(item.get("raw_action", []), dtype=np.float64)
        if pressure.ndim != 2 or pressure.shape[1] != 2 or len(time) != len(pressure):
            raise ValueError(f"trajectory {trajectory_id}: invalid time/pressure shape")
        checked = pressure[min(start, len(pressure) - 1) :]
        rate = np.gradient(pressure, dt, axis=0)
        checked_rate = rate[min(start, len(rate) - 1) :]
        meta = dict(item.get("metadata", {}))
        cycles = _as_pair(meta.get("requested_cycles", [float("nan"), float("nan")]), "requested_cycles")
        phase = _as_pair(meta.get("phase_rad", [float("nan"), float("nan")]), "phase_rad")
        pos_timing = _schedule_timing(meta, "pos")
        neg_timing = _schedule_timing(meta, "neg")
        oracle_row = oracle.get(trajectory_id)
        oracle_pass = oracle_row is not None and max(oracle_row.values()) <= 1e-6
        pressure_pass = bool(
            np.min(checked[:, 0]) >= pos_band[0]
            and np.max(checked[:, 0]) <= pos_band[1]
            and np.min(checked[:, 1]) >= neg_band[0]
            and np.max(checked[:, 1]) <= neg_band[1]
        )
        rate_pass = bool(np.max(np.abs(checked_rate[:, 0])) <= rate_pos_max and np.max(np.abs(checked_rate[:, 1])) <= rate_neg_max)
        phase_difference = _wrapped_phase_difference(*phase)
        independent_schedule = bool(
            pos_timing["count"] > 0
            and neg_timing["count"] > 0
            and (
                pos_timing["count"] != neg_timing["count"]
                or not _same_timing(pos_timing["boundaries"], neg_timing["boundaries"])
                or not _same_timing(pos_timing["transition_values"], neg_timing["transition_values"])
                or not _same_timing(pos_timing["hold_values"], neg_timing["hold_values"])
            )
        )
        row = {
            "trajectory_index": index,
            "trajectory_id": trajectory_id,
            "duration_sec": float(time[-1] - time[0]),
            "requested_cycles_pos": cycles[0],
            "requested_cycles_neg": cycles[1],
            "phase_pos_rad": phase[0],
            "phase_neg_rad": phase[1],
            "phase_difference_rad": phase_difference,
            "pos_transition_count": pos_timing["count"],
            "neg_transition_count": neg_timing["count"],
            "pos_transition_sec_mean": pos_timing["transition_mean"],
            "neg_transition_sec_mean": neg_timing["transition_mean"],
            "pos_hold_sec_mean": pos_timing["hold_mean"],
            "neg_hold_sec_mean": neg_timing["hold_mean"],
            "pressure_turns_pos": int(meta.get("pressure_turns", [0, 0])[0]),
            "pressure_turns_neg": int(meta.get("pressure_turns", [0, 0])[1]),
            "pos_min_kpa_after_startup": float(np.min(checked[:, 0])),
            "pos_max_kpa_after_startup": float(np.max(checked[:, 0])),
            "neg_min_kpa_after_startup": float(np.min(checked[:, 1])),
            "neg_max_kpa_after_startup": float(np.max(checked[:, 1])),
            "pos_span_kpa": float(np.ptp(checked[:, 0])),
            "neg_span_kpa": float(np.ptp(checked[:, 1])),
            "pos_rate_abs_max_kpa_s": float(np.max(np.abs(checked_rate[:, 0]))),
            "neg_rate_abs_max_kpa_s": float(np.max(np.abs(checked_rate[:, 1]))),
            "raw_pos_min": float(np.min(action[:, 0])) if action.size else float("nan"),
            "raw_pos_max": float(np.max(action[:, 0])) if action.size else float("nan"),
            "raw_neg_min": float(np.min(action[:, 1])) if action.size else float("nan"),
            "raw_neg_max": float(np.max(action[:, 1])) if action.size else float("nan"),
            "pressure_band_pass": pressure_pass,
            "rate_limit_pass": rate_pass,
            "oracle_replay_pass": oracle_pass,
            "joint_feasible_pass": bool(action.size and oracle_pass),
            "independent_timing_parameterized": bool(
                independent_schedule
                or (math.isfinite(phase_difference) and (abs(cycles[0] - cycles[1]) > 1e-6 or phase_difference > 1e-6))
            ),
            "oracle_pos_rmse": None if oracle_row is None else oracle_row["pos_rmse"],
            "oracle_neg_rmse": None if oracle_row is None else oracle_row["neg_rmse"],
        }
        row["all_checks_pass"] = bool(pressure_pass and rate_pass and row["joint_feasible_pass"])
        rows.append(row)

    summary = {
        "reference_profile": str(profile_path),
        "trajectory_count": len(rows),
        "frequency_hz": frequency,
        "startup_filter_sec": float(design.get("rate_filter_ignore_initial_sec", 0.0)),
        "configured_pressure_band_kpa": {"pos": list(pos_band), "neg": list(neg_band)},
        "configured_rate_abs_max_kpa_s": {"pos": rate_pos_max, "neg": rate_neg_max},
        "all_checks_pass": bool(all(row["all_checks_pass"] for row in rows)),
        "pass_counts": {
            "pressure_band": int(sum(row["pressure_band_pass"] for row in rows)),
            "rate_limit": int(sum(row["rate_limit_pass"] for row in rows)),
            "oracle_replay_joint_feasibility": int(sum(row["joint_feasible_pass"] for row in rows)),
            "independent_timing_parameterized": int(sum(row["independent_timing_parameterized"] for row in rows)),
        },
        "coverage": {
            "pos_transition_count": [int(min(row["pos_transition_count"] for row in rows)), int(max(row["pos_transition_count"] for row in rows))],
            "neg_transition_count": [int(min(row["neg_transition_count"] for row in rows)), int(max(row["neg_transition_count"] for row in rows))],
            "pos_hold_sec_mean": [float(np.nanmin([row["pos_hold_sec_mean"] for row in rows])), float(np.nanmax([row["pos_hold_sec_mean"] for row in rows]))],
            "neg_hold_sec_mean": [float(np.nanmin([row["neg_hold_sec_mean"] for row in rows])), float(np.nanmax([row["neg_hold_sec_mean"] for row in rows]))],
            "pos_span_kpa": [float(min(row["pos_span_kpa"] for row in rows)), float(max(row["pos_span_kpa"] for row in rows))],
            "neg_span_kpa": [float(min(row["neg_span_kpa"] for row in rows)), float(max(row["neg_span_kpa"] for row in rows))],
        },
    }
    return rows, summary


def _plot_pressure_grid(trajectories: list[dict[str, Any]], rows: list[dict[str, Any]], profile: dict[str, Any], output: Path) -> None:
    design = profile["design"]
    pos_band = _as_pair(design["pressure_filter_kpa"]["pos"], "POS pressure filter")
    neg_band = _as_pair(design["pressure_filter_kpa"]["neg"], "NEG pressure filter")
    fig, axes = plt.subplots(len(trajectories), 2, figsize=(16, max(16, 2.15 * len(trajectories))), sharex=True)
    axes = np.atleast_2d(axes)
    for row, item, axis_pair in zip(rows, trajectories, axes):
        time = np.asarray(item["time"], dtype=np.float64)
        pressure = np.asarray(item["pressure"], dtype=np.float64)
        pos_axis, neg_axis = axis_pair
        pos_axis.plot(time, pressure[:, 0], color="red", linewidth=1.4)
        neg_axis.plot(time, pressure[:, 1], color="blue", linewidth=1.4)
        for axis, band, color, label in ((pos_axis, pos_band, "red", "POS"), (neg_axis, neg_band, "blue", "NEG")):
            axis.axhspan(*band, color=color, alpha=0.07)
            axis.axhline(band[0], color=color, linestyle="--", alpha=0.4, linewidth=0.8)
            axis.axhline(band[1], color=color, linestyle="--", alpha=0.4, linewidth=0.8)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(f"{label} [kPa]")
        pos_axis.set_title(
            f"traj {row['trajectory_id']} | POS transitions={row['pos_transition_count']}, hold={row['pos_hold_sec_mean']:.2f}s, turns={row['pressure_turns_pos']}, span={row['pos_span_kpa']:.1f}"
        )
        neg_axis.set_title(
            f"NEG transitions={row['neg_transition_count']}, hold={row['neg_hold_sec_mean']:.2f}s, turns={row['pressure_turns_neg']}, span={row['neg_span_kpa']:.1f}"
        )
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle("lib2 optimized reset reference bank — POS/NEG shown separately", y=0.998, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.993))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row["trajectory_id"]) for row in rows]
    index = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].bar(index - 0.18, [row["pos_span_kpa"] for row in rows], 0.36, color="red", label="POS")
    axes[0, 0].bar(index + 0.18, [row["neg_span_kpa"] for row in rows], 0.36, color="blue", label="NEG")
    axes[0, 0].set_title("Pressure span after startup")
    axes[0, 0].set_ylabel("span [kPa]")
    axes[0, 0].legend()
    axes[0, 1].plot(index, [row["pos_transition_count"] for row in rows], "o-", color="red", label="POS transitions")
    axes[0, 1].plot(index, [row["neg_transition_count"] for row in rows], "o-", color="blue", label="NEG transitions")
    axes[0, 1].plot(index, [row["pos_hold_sec_mean"] for row in rows], "o--", color="darkred", label="POS mean hold")
    axes[0, 1].plot(index, [row["neg_hold_sec_mean"] for row in rows], "o--", color="navy", label="NEG mean hold")
    axes[0, 1].set_title("Independently sampled transition / hold timing")
    axes[0, 1].set_ylabel("count / sec")
    axes[0, 1].legend()
    axes[1, 0].bar(index - 0.18, [row["pos_rate_abs_max_kpa_s"] for row in rows], 0.36, color="red", label="POS")
    axes[1, 0].bar(index + 0.18, [row["neg_rate_abs_max_kpa_s"] for row in rows], 0.36, color="blue", label="NEG")
    axes[1, 0].set_title("Maximum reference |dP/dt| after startup")
    axes[1, 0].set_ylabel("kPa/s")
    axes[1, 0].legend()
    axes[1, 1].scatter([row["pos_span_kpa"] for row in rows], [row["neg_span_kpa"] for row in rows], c=index, cmap="turbo", s=70)
    for row in rows:
        axes[1, 1].annotate(str(row["trajectory_id"]), (row["pos_span_kpa"], row["neg_span_kpa"]), xytext=(4, 4), textcoords="offset points")
    axes[1, 1].set_title("Joint pressure-span coverage")
    axes[1, 1].set_xlabel("POS span [kPa]")
    axes[1, 1].set_ylabel("NEG span [kPa]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_xticks(index, labels)
        axis.set_xlabel("trajectory id")
    axes[0, 1].set_xticks(index, labels)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--oracle-metrics", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    profile, trajectories = _load_profile(args.reference_profile)
    oracle = _load_oracle(args.oracle_metrics)
    rows, summary = _validate(profile, trajectories, oracle, args.reference_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation.csv", rows)
    with (args.output_dir / "validation.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    _plot_pressure_grid(trajectories, rows, profile, args.output_dir / "pressure_grid.png")
    _plot_coverage(rows, args.output_dir / "coverage.png")
    print(f"[ INFO] validation: {args.output_dir / 'validation.yaml'}")
    print(f"[ INFO] pressure grid: {args.output_dir / 'pressure_grid.png'}")
    print(f"[ INFO] all checks pass: {summary['all_checks_pass']}")


if __name__ == "__main__":
    main()
