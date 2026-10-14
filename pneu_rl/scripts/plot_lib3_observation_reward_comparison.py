#!/usr/bin/env python3
"""Plot the three-arm SAC/history/rate inductive-bias screen for lib3.

The three runs use the same 512-step training protocol and base reward:
current-observation SAC, History SAC, and History SAC + explicit pressure rate.
The accompanying Markdown records the single-seed protocol and its limits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    "SAC": "260806_lib3_capref_512_pure_sac_seed0",
    "History SAC": "260810_lib3_ablation_profile_hist_base_seed0",
    "History SAC + rate": "260810_lib3_ablation_profile_hist_rate_seed0",
}
COLORS = {"SAC": "#666666", "History SAC": "#3B6EA5", "History SAC + rate": "#E24A33"}


def load_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    episode = np.asarray([int(row["epi"]) for row in rows], dtype=np.int64)
    reward = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    pos_rmse = np.asarray([float(row["pos_rmse"]) for row in rows], dtype=np.float64)
    neg_rmse = np.asarray([float(row["neg_rmse"]) for row in rows], dtype=np.float64)
    combined_rmse = np.sqrt((pos_rmse**2 + neg_rmse**2) / 2.0)
    return episode, reward, combined_rmse


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean with a partial window at the beginning.

    This keeps the visible learning curve anchored at episode 1.  From episode
    ``window`` onward it is exactly the usual fixed-window moving mean.
    """
    cumulative = np.cumsum(values, dtype=np.float64)
    cumulative[window:] -= cumulative[:-window]
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return cumulative / counts


def first_threshold_episode(episodes: np.ndarray, values: np.ndarray, threshold: float) -> int | str:
    hit = np.flatnonzero(values <= threshold)
    return int(episodes[hit[0]]) if hit.size else "not_reached"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("pneu_rl/models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "pneu_rl/exp/lib3_0830_history_plateau/"
            "sac_history_rate_learning_dynamics.png"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "pneu_rl/exp/lib3_0830_history_plateau/"
            "sac_history_rate_learning_summary.csv"
        ),
    )
    parser.add_argument("--rolling-window", type=int, default=50)
    args = parser.parse_args()

    if args.rolling_window < 2:
        parser.error("--rolling-window must be at least 2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, reward_axis = plt.subplots(figsize=(12.5, 6.3))
    figure.suptitle(
        "Effect of history and explicit pressure-rate coordinates on SAC learning",
        fontsize=19,
        y=0.985,
    )
    summary_rows: list[dict[str, str | float | int]] = []

    for label, model in MODELS.items():
        episode, reward, combined_rmse = load_diagnostics(
            args.model_root / model / "train_diagnostics.csv"
        )
        window = min(args.rolling_window, len(reward))
        smoothed_reward = rolling_mean(reward, window)
        smoothed_rmse = rolling_mean(combined_rmse, window)
        smooth_episode = episode
        color = COLORS[label]

        reward_axis.plot(episode, reward, color=color, alpha=0.08, linewidth=0.7)
        reward_axis.plot(smooth_episode, smoothed_reward, color=color, linewidth=2.5, label=label)
        summary_rows.append(
            {
                "method": label,
                "rolling_window": window,
                "episode_to_2kpa_rolling_rmse": first_threshold_episode(smooth_episode, smoothed_rmse, 2.0),
                "episode_to_1_5kpa_rolling_rmse": first_threshold_episode(smooth_episode, smoothed_rmse, 1.5),
                "final_50_episode_mean_reward": float(smoothed_reward[-1]),
                "final_50_episode_mean_combined_rmse_kpa": float(smoothed_rmse[-1]),
            }
        )

    reward_axis.set_title("Training return (50-episode trailing mean; higher is better)")
    reward_axis.set_ylabel("Episode reward")
    reward_axis.set_xlim(1, 1000)
    # The 50-episode SAC mean reaches about -8.5k early in training.  Keep that
    # starting region visible instead of clipping it at the old -4k display limit.
    reward_axis.set_ylim(-10000, 0)
    reward_axis.legend(loc="lower right", frameon=True)
    reward_axis.set_xlabel("Training episode")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved: {args.output}")
    print(f"saved: {args.summary_output}")


if __name__ == "__main__":
    main()
