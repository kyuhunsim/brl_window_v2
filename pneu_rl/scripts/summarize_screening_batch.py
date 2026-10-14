"""Summarize comparable screening outputs without relying on raw reward alone."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_row(screen_dir: Path) -> dict[str, Any]:
    with (screen_dir / "metrics.yaml").open(encoding="utf-8") as handle:
        metrics = dict(yaml.safe_load(handle))
    perturbation_path = screen_dir / "local_action_perturbation.yaml"
    if perturbation_path.is_file():
        with perturbation_path.open(encoding="utf-8") as handle:
            perturbation = yaml.safe_load(handle) or {}
        summary = perturbation.get("summary", {})
        metrics["local_improved_point_ratio"] = summary.get("improved_point_ratio")
        metrics["local_strongly_improved_ratio"] = summary.get("strongly_improved_ratio")
        metrics["local_best_improvement_median"] = summary.get("best_improvement_median")
        metrics["local_best_improvement_max"] = summary.get("best_improvement_max")
    else:
        metrics["local_improved_point_ratio"] = None
        metrics["local_strongly_improved_ratio"] = None
        metrics["local_best_improvement_median"] = None
        metrics["local_best_improvement_max"] = None
    opportunity_path = screen_dir / "control_opportunity.yaml"
    if opportunity_path.is_file():
        with opportunity_path.open(encoding="utf-8") as handle:
            opportunity = yaml.safe_load(handle) or {}
        pos_deficit = opportunity.get("pos", {}).get("deficit_needs_lower_u_pos", {})
        neg_deficit = opportunity.get("neg", {}).get("deficit_needs_higher_u_neg", {})
        metrics["pos_deficit_helpful_headroom_median"] = pos_deficit.get("margin_median")
        metrics["pos_deficit_helpful_near_bound_ratio"] = pos_deficit.get("near_bound_ratio_margin_lt_0.10")
        metrics["neg_deficit_helpful_headroom_median"] = neg_deficit.get("margin_median")
        metrics["neg_deficit_helpful_near_bound_ratio"] = neg_deficit.get("near_bound_ratio_margin_lt_0.10")
        metrics["simultaneous_pos_neg_deficit_ratio"] = opportunity.get("joint_error_geometry", {}).get(
            "simultaneous_pos_neg_deficit_ratio"
        )
    else:
        metrics["pos_deficit_helpful_headroom_median"] = None
        metrics["pos_deficit_helpful_near_bound_ratio"] = None
        metrics["neg_deficit_helpful_headroom_median"] = None
        metrics["neg_deficit_helpful_near_bound_ratio"] = None
        metrics["simultaneous_pos_neg_deficit_ratio"] = None
    metrics["screen_dir"] = str(screen_dir)
    return metrics


def run(args: argparse.Namespace) -> None:
    rows = [load_row(Path(value)) for value in args.screen_dir]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "variant", "control_mode", "training_pid_enabled", "episode_carry_state",
        "reference_profile_source", "reference_match_scope", "reference_bank_member_index",
        "reference_bank_member_id", "steps", "duration_sec", "reward_per_step",
        "pos_rmse", "neg_rmse", "pos_abs_error_p95", "neg_abs_error_p95",
        "pos_bias", "neg_bias", "pos_success_ratio", "neg_success_ratio",
        "pos_ctrl_low_fraction", "neg_ctrl_low_fraction", "pos_ctrl_high_fraction", "neg_ctrl_high_fraction",
        "pos_ctrl_mean_headroom_low", "neg_ctrl_mean_headroom_low",
        "pos_ctrl_mean_headroom_high", "neg_ctrl_mean_headroom_high",
        "pos_ctrl_delta", "neg_ctrl_delta", "pos_pressure_sign_change_per_sec",
        "neg_pressure_sign_change_per_sec", "rate_filter_evaluated",
        "joint_ref_rate_violation_ratio", "local_improved_point_ratio",
        "local_strongly_improved_ratio", "local_best_improvement_median",
        "local_best_improvement_max", "pos_deficit_helpful_headroom_median",
        "pos_deficit_helpful_near_bound_ratio", "neg_deficit_helpful_headroom_median",
        "neg_deficit_helpful_near_bound_ratio", "simultaneous_pos_neg_deficit_ratio",
        "screen_dir",
    ]
    rows_for_csv = [{field: row.get(field) for field in fields} for row in rows]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_for_csv)
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(rows_for_csv, handle, sort_keys=False)

    labels = []
    for row in rows:
        label = str(row.get("model", Path(row["screen_dir"]).parent.name))
        member = row.get("reference_bank_member_index")
        if member is not None:
            label = f"{label}\nmember {int(member):02d}"
        labels.append(label)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.36
    axes[0, 0].bar(x - width / 2, [row.get("pos_rmse", np.nan) for row in rows], width, label="POS")
    axes[0, 0].bar(x + width / 2, [row.get("neg_rmse", np.nan) for row in rows], width, label="NEG")
    axes[0, 0].set_title("Pressure RMSE")
    axes[0, 0].set_ylabel("kPa")
    axes[0, 0].legend()
    axes[0, 1].bar(x, [row.get("reward_per_step", np.nan) for row in rows], color="gray")
    axes[0, 1].set_title("Reward per step")
    axes[0, 1].set_ylabel("mean reward")
    axes[1, 0].bar(x - width / 2, [row.get("pos_ctrl_high_fraction", np.nan) for row in rows], width, label="POS high")
    axes[1, 0].bar(x + width / 2, [row.get("neg_ctrl_high_fraction", np.nan) for row in rows], width, label="NEG high")
    axes[1, 0].set_title("High-action saturation")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].legend()
    axes[1, 1].bar(
        x,
        [np.nan if row.get("local_best_improvement_median") is None else row.get("local_best_improvement_median") for row in rows],
        color="darkgreen",
    )
    axes[1, 1].set_title("Median best local improvement")
    axes[1, 1].set_ylabel("short-horizon cost improvement")
    for axis in axes.flat:
        axis.grid(True, axis="y", color="lightgray")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] wrote {output_dir / 'summary.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", action="append", required=True, help="screen output directory; repeat this option")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
