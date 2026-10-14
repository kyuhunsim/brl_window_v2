#!/usr/bin/env python3
"""Plot comparable training diagnostics for the lib4 R3 screening matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_LABELS = {
    "260823_lib4_r3_direct_seed0": "Direct (870 epi)",
    "260823_lib4_r3_history_seed0": "History (missing)",
    "260823_lib4_r3_history_rate_seed0": "History + rate",
    "260823_lib4_r3_history_reserve_seed0": "History + reserve",
    "260823_lib4_r3_history_rate_reserve_seed0": "History + rate+reserve",
    "260823_lib4_r3_history_rate_reserve_k2_seed0": "History + rate+reserve K=2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    available: list[tuple[str, str, pd.DataFrame]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    action_delta_columns = (
        "u_ch_pos_delta",
        "u_ch_neg_delta",
        "u_pos_in_delta",
        "u_pos_out_delta",
        "u_neg_in_delta",
        "u_neg_out_delta",
    )
    for model, label in MODEL_LABELS.items():
        path = ROOT / "pneu_rl" / "models" / model / "train_diagnostics.csv"
        if not path.is_file():
            summary_rows.append({"model": model, "label": label, "episodes": 0, "status": "missing"})
            continue
        data = pd.read_csv(path)
        data["pressure_rmse_mean"] = 0.5 * (data["pos_rmse"] + data["neg_rmse"])
        data["action_delta_mean"] = data[list(action_delta_columns)].mean(axis=1)
        available.append((model, label, data))
        tail = data.tail(100)
        summary_rows.append(
            {
                "model": model,
                "label": label,
                "episodes": int(len(data)),
                "status": "complete" if len(data) == 1000 else "incomplete",
                "reward_final100": float(tail["reward"].mean()),
                "pos_rmse_final100": float(tail["pos_rmse"].mean()),
                "neg_rmse_final100": float(tail["neg_rmse"].mean()),
                "disp_rmse_final100": float(tail["disp_rmse"].mean()),
                "pos_bias_final100": float(tail["pos_bias"].mean()),
                "neg_bias_final100": float(tail["neg_bias"].mean()),
                "disp_bias_final100": float(tail["disp_bias"].mean()),
                "action_delta_final100": float(tail["action_delta_mean"].mean()),
                "disp_direction_final100": float(tail["disp_direction_agreement"].mean()),
            }
        )

    with (output_dir / "training_final100.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in summary_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    figure, axes = plt.subplots(4, 1, figsize=(13, 14), constrained_layout=True)
    panels = (
        ("reward", "Episode reward", None),
        ("pressure_rmse_mean", "Mean pressure RMSE [kPa]", (0.0, 8.0)),
        ("disp_rmse", "Displacement RMSE [mm]", (0.0, 5.0)),
        ("action_delta_mean", "Mean |Δu|", (0.0, None)),
    )
    for _, label, data in available:
        episode = data["epi"].to_numpy(dtype=np.float64)
        for axis, (key, ylabel, ylim) in zip(axes, panels):
            smooth = data[key].rolling(25, min_periods=1).mean()
            axis.plot(episode, smooth, linewidth=1.6, label=label)
            axis.set_xlabel("Episode")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.35)
            if ylim is not None:
                axis.set_ylim(*ylim)
    for axis in axes:
        axis.legend(loc="best", fontsize=9)
    figure.suptitle("lib4 R3 seed-0 training diagnostics — rolling mean 25 episodes")
    figure.savefig(output_dir / "training_comparison.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
