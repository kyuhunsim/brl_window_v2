#!/usr/bin/env python3
"""Plot training reward convergence and post-convergence control diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = [
    "260810_lib3_ablation_profile_hist_base_seed0",
    "260810_lib3_ablation_profile_hist_rate_reserve_seed0",
    "260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0",
]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty diagnostics file: {path}")
    keys = rows[0].keys()
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in keys
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    window = max(2, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def tail_stats(data: dict[str, np.ndarray]) -> dict[str, float | int | str]:
    epi = data["epi"].astype(int)
    n = len(epi)
    policy_start = int(np.searchsorted(epi, 11))
    policy = slice(policy_start, n)
    tail_n = max(10, min(20, n - policy_start))
    early_end = min(policy_start + tail_n, n)
    early = slice(policy_start, early_end)
    tail = slice(n - tail_n, n)

    reward = data["reward"]
    roll_window = max(5, min(10, n // 4))
    roll = moving_average(reward, roll_window)
    # The last-window linear slope is a descriptive convergence diagnostic,
    # not a claim that the policy has reached a global optimum.
    x = epi[tail].astype(np.float64)
    y = roll[tail]
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")

    def mean(name: str, section: slice) -> float:
        return float(np.nanmean(data[name][section]))

    action_delta_cols = [
        "u_ch_pos_delta", "u_ch_neg_delta", "u_pos_in_delta",
        "u_pos_out_delta", "u_neg_in_delta", "u_neg_out_delta",
    ]
    action_delta_early = float(np.nanmean(np.column_stack([data[c][early] for c in action_delta_cols])))
    action_delta_tail = float(np.nanmean(np.column_stack([data[c][tail] for c in action_delta_cols])))
    sign_early = float(np.nanmean(np.column_stack([
        data["act_pos_sign_change_per_sec"][early],
        data["act_neg_sign_change_per_sec"][early],
    ])))
    sign_tail = float(np.nanmean(np.column_stack([
        data["act_pos_sign_change_per_sec"][tail],
        data["act_neg_sign_change_per_sec"][tail],
    ])))

    return {
        "episodes": n,
        "last_episode": int(epi[-1]),
        "roll_window": roll_window,
        "best_reward": float(np.max(reward[policy])),
        "best_reward_episode": int(epi[policy][np.argmax(reward[policy])]),
        "early_reward": mean("reward", early),
        "tail_reward": mean("reward", tail),
        "tail_reward_std": float(np.nanstd(reward[tail])),
        "tail_roll_slope_per_episode": slope,
        "early_pos_rmse": mean("pos_rmse", early),
        "tail_pos_rmse": mean("pos_rmse", tail),
        "early_neg_rmse": mean("neg_rmse", early),
        "tail_neg_rmse": mean("neg_rmse", tail),
        "early_action_delta": action_delta_early,
        "tail_action_delta": action_delta_tail,
        "early_sign_change": sign_early,
        "tail_sign_change": sign_tail,
        "tail_ch_pos_max": float(np.nanmax(data["ch_pos_max"][tail])),
        "tail_ch_neg_min": float(np.nanmin(data["ch_neg_min"][tail])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("pneu_rl/models"))
    parser.add_argument("--output-root", type=Path, default=Path("pneu_rl/exp/lib3_0810_reward_analysis"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[str, dict[str, np.ndarray]]] = []
    summaries: list[dict[str, float | int | str]] = []
    for model in args.models:
        path = args.model_root / model / "train_diagnostics.csv"
        if not path.exists():
            print(f"MISSING: {model}")
            continue
        data = load_csv(path)
        loaded.append((model, data))
        summary = {"model": model, **tail_stats(data)}
        summaries.append(summary)

    if not loaded:
        raise SystemExit("no train_diagnostics.csv found")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (model, data) in enumerate(loaded):
        color = colors[index % len(colors)]
        epi = data["epi"]
        reward = data["reward"]
        roll = moving_average(reward, max(5, min(10, len(reward) // 4)))
        label = model.replace("260810_lib3_ablation_profile_hist_", "")
        axes[0].plot(epi, reward, color=color, alpha=0.25)
        axes[0].plot(epi, roll, color=color, linewidth=2, label=label)
        axes[1].plot(epi, data["pos_rmse"], color=color, linestyle="-", alpha=0.8)
        axes[1].plot(epi, data["neg_rmse"], color=color, linestyle="--", alpha=0.8)
        action_delta = np.nanmean(np.column_stack([
            data["u_ch_pos_delta"], data["u_ch_neg_delta"],
            data["u_pos_in_delta"], data["u_pos_out_delta"],
            data["u_neg_in_delta"], data["u_neg_out_delta"],
        ]), axis=1)
        sign_change = np.nanmean(np.column_stack([
            data["act_pos_sign_change_per_sec"],
            data["act_neg_sign_change_per_sec"],
        ]), axis=1)
        axes[2].plot(epi, action_delta, color=color, linewidth=1.8, label=label)
        axes[3].plot(epi, sign_change, color=color, linewidth=1.8, label=label)

    axes[0].axvline(10.5, color="black", linestyle=":", linewidth=1, label="warm-up end")
    axes[0].set_title("lib3 completed ablations: episode reward and rolling mean")
    axes[0].set_ylabel("episode reward")
    axes[0].legend()
    axes[1].set_title("Episode tracking RMSE (solid=POS, dashed=NEG)")
    axes[1].set_ylabel("RMSE [kPa]")
    axes[2].set_title("Mean |Δu| across six physical commands")
    axes[2].set_ylabel("mean |Δu|")
    axes[2].legend()
    axes[3].set_title("actuator pressure sign-change rate: vibration proxy")
    axes[3].set_ylabel("sign changes/s")
    axes[3].set_xlabel("training episode")
    axes[3].legend()
    fig.tight_layout()
    plot_path = args.output_root / "reward_convergence_and_post_convergence.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    fields = sorted({key for row in summaries for key in row})
    summary_path = args.output_root / "reward_convergence_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"saved: {plot_path}")
    print(f"saved: {summary_path}")
    for row in summaries:
        print(
            f"{row['model']}: episodes={row['episodes']} "
            f"early_reward={row['early_reward']:.1f} "
            f"tail_reward={row['tail_reward']:.1f} "
            f"tail_slope={row['tail_roll_slope_per_episode']:.2f} "
            f"RMSE_POS={row['early_pos_rmse']:.2f}->{row['tail_pos_rmse']:.2f} "
            f"RMSE_NEG={row['early_neg_rmse']:.2f}->{row['tail_neg_rmse']:.2f} "
            f"|du|={row['early_action_delta']:.4f}->{row['tail_action_delta']:.4f} "
            f"sign={row['early_sign_change']:.2f}->{row['tail_sign_change']:.2f}"
        )


if __name__ == "__main__":
    main()
