#!/usr/bin/env python3
"""Create publication-style figures for the lib3 H/HR x reward experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ARMS = {
    "H-base": "260831_lib3_histreward2x2_r025_H_base_seed0",
    "H-smooth": "260831_lib3_histreward2x2_r025_H_smooth_seed0",
    "HR-base": "260831_lib3_histreward2x2_r025_HR_base_seed0",
    "HR-smooth": "260831_lib3_histreward2x2_r025_HR_smooth_seed0",
}

COLORS = {
    "H-base": "#4C78A8",
    "H-smooth": "#72B7B2",
    "HR-base": "#E45756",
    "HR-smooth": "#F2A541",
}


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def metric_arrays(summary: dict, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in summary["episodes"]], dtype=float)


def paired_change(new: np.ndarray, old: np.ndarray) -> tuple[float, float]:
    difference = new - old
    percent = 100.0 * (np.mean(new) / np.mean(old) - 1.0)
    return float(np.mean(difference)), float(percent)


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_end = 0
    run_start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            if index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            run_start = None
    return best_start, best_end


def build_metrics_table(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, summary in summaries.items():
        mean = summary["mean"]
        std = summary["std"]
        combined_rmse = np.sqrt(
            (
                metric_arrays(summary, "pos_rmse_kpa") ** 2
                + metric_arrays(summary, "neg_rmse_kpa") ** 2
            )
            / 2.0
        )
        combined_rate = np.sqrt(
            (
                metric_arrays(summary, "plateau_pos_pressure_rate_rms_kpa_s") ** 2
                + metric_arrays(summary, "plateau_neg_pressure_rate_rms_kpa_s") ** 2
            )
            / 2.0
        )
        rows.append(
            {
                "arm": label,
                "pos_rmse_mean_kpa": mean["pos_rmse_kpa"],
                "pos_rmse_std_kpa": std["pos_rmse_kpa"],
                "neg_rmse_mean_kpa": mean["neg_rmse_kpa"],
                "neg_rmse_std_kpa": std["neg_rmse_kpa"],
                "combined_rmse_mean_kpa": float(np.mean(combined_rmse)),
                "combined_rmse_std_kpa": float(np.std(combined_rmse)),
                "plateau_pos_rate_rms_mean_kpa_s": mean[
                    "plateau_pos_pressure_rate_rms_kpa_s"
                ],
                "plateau_pos_rate_rms_std_kpa_s": std[
                    "plateau_pos_pressure_rate_rms_kpa_s"
                ],
                "plateau_neg_rate_rms_mean_kpa_s": mean[
                    "plateau_neg_pressure_rate_rms_kpa_s"
                ],
                "plateau_neg_rate_rms_std_kpa_s": std[
                    "plateau_neg_pressure_rate_rms_kpa_s"
                ],
                "combined_plateau_rate_rms_mean_kpa_s": float(
                    np.mean(combined_rate)
                ),
                "plateau_action_delta_rms_mean": mean[
                    "plateau_action_delta_rms"
                ],
                "plateau_action_delta_rms_std": std[
                    "plateau_action_delta_rms"
                ],
                "positive_reserve_ge_10_fraction_mean": mean[
                    "positive_reserve_ge_10kpa_fraction"
                ],
                "negative_reserve_ge_10_fraction_mean": mean[
                    "negative_reserve_ge_10kpa_fraction"
                ],
            }
        )
    return pd.DataFrame(rows)


def plot_summary(
    summaries: dict[str, dict],
    model_root: Path,
    table: pd.DataFrame,
    output: Path,
    rolling_window: int,
) -> None:
    labels = list(ARMS)
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))

    ax = axes[0, 0]
    for label, model in ARMS.items():
        train = pd.read_csv(model_root / model / "train_diagnostics.csv")
        combined = np.sqrt((train["pos_rmse"] ** 2 + train["neg_rmse"] ** 2) / 2)
        smooth = rolling_mean(combined.to_numpy(), rolling_window)
        ax.plot(
            train["epi"],
            smooth,
            color=COLORS[label],
            linewidth=2.0,
            label=label,
        )
    ax.set_title(f"Training tracking error ({rolling_window}-episode mean)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Combined RMSE (kPa)")
    ax.set_xlim(1, 2000)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, ncol=2)

    ax = axes[0, 1]
    pos = table["pos_rmse_mean_kpa"].to_numpy()
    neg = table["neg_rmse_mean_kpa"].to_numpy()
    ax.bar(
        x - width / 2,
        pos,
        width,
        yerr=table["pos_rmse_std_kpa"],
        label="POS",
        color="#4C78A8",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        neg,
        width,
        yerr=table["neg_rmse_std_kpa"],
        label="NEG",
        color="#E45756",
        capsize=3,
    )
    ax.set_title("Held-out full-bank tracking")
    ax.set_ylabel("RMSE (kPa)")
    ax.set_xticks(x, labels, rotation=18)
    ax.legend(frameon=True)

    ax = axes[0, 2]
    pos_rate = table["plateau_pos_rate_rms_mean_kpa_s"].to_numpy()
    neg_rate = table["plateau_neg_rate_rms_mean_kpa_s"].to_numpy()
    ax.bar(
        x - width / 2,
        pos_rate,
        width,
        yerr=table["plateau_pos_rate_rms_std_kpa_s"],
        label="POS",
        color="#4C78A8",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        neg_rate,
        width,
        yerr=table["plateau_neg_rate_rms_std_kpa_s"],
        label="NEG",
        color="#E45756",
        capsize=3,
    )
    ax.set_title("Plateau pressure motion")
    ax.set_ylabel(r"Pressure-rate RMS (kPa s$^{-1}$)")
    ax.set_xticks(x, labels, rotation=18)
    ax.legend(frameon=True)

    ax = axes[1, 0]
    action = table["plateau_action_delta_rms_mean"].to_numpy()
    ax.bar(
        x,
        action,
        yerr=table["plateau_action_delta_rms_std"],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    ax.set_title("Plateau command variation")
    ax.set_ylabel(r"RMS of $\Delta u$ (physical command)")
    ax.set_xticks(x, labels, rotation=18)

    ax = axes[1, 1]
    pos_reserve = 100.0 * table["positive_reserve_ge_10_fraction_mean"].to_numpy()
    neg_reserve = 100.0 * table["negative_reserve_ge_10_fraction_mean"].to_numpy()
    ax.bar(
        x - width / 2,
        pos_reserve,
        width,
        label="POS",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        neg_reserve,
        width,
        label="NEG",
        color="#E45756",
    )
    ax.set_title("Pressure-reserve availability")
    ax.set_ylabel("Samples with reserve $\geq$ 10 kPa (%)")
    ax.set_ylim(80, 100.5)
    ax.set_xticks(x, labels, rotation=18)
    ax.legend(frameon=True)

    ax = axes[1, 2]
    for _, row in table.iterrows():
        label = row["arm"]
        ax.scatter(
            row["combined_rmse_mean_kpa"],
            row["combined_plateau_rate_rms_mean_kpa_s"],
            s=95,
            color=COLORS[label],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            label,
            (
                row["combined_rmse_mean_kpa"],
                row["combined_plateau_rate_rms_mean_kpa_s"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )
    ax.set_title("Tracking–motion trade-off (lower is better)")
    ax.set_xlabel("Combined RMSE (kPa)")
    ax.set_ylabel(r"Combined plateau-rate RMS (kPa s$^{-1}$)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    figure.suptitle(
        "lib3 observation × reward ablation (seed 0, final checkpoints, 16 references)",
        fontsize=17,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def choose_representative_member(
    summaries: dict[str, dict],
) -> int:
    per_member = []
    for index in range(16):
        values = []
        for summary in summaries.values():
            row = summary["episodes"][index]
            values.append(
                np.sqrt((row["pos_rmse_kpa"] ** 2 + row["neg_rmse_kpa"] ** 2) / 2)
            )
        per_member.append(float(np.mean(values)))
    target = float(np.median(per_member))
    return int(np.argmin(np.abs(np.asarray(per_member) - target)))


def plot_trajectory(
    evaluation_root: Path,
    member: int,
    output: Path,
) -> None:
    data = {
        label: pd.read_csv(evaluation_root / model / f"member_{member:02d}.csv")
        for label, model in ARMS.items()
    }
    base = next(iter(data.values()))
    time = base["time"].to_numpy()
    dt = float(np.median(np.diff(time)))
    ref_pos = base["ref_pos"].to_numpy()
    ref_neg = base["ref_neg"].to_numpy()
    rate_pos = np.gradient(ref_pos, dt)
    rate_neg = np.gradient(ref_neg, dt)
    plateau = (time >= 1.0) & (np.abs(rate_pos) < 1.0) & (np.abs(rate_neg) < 0.5)
    start, end = longest_true_run(plateau)
    if end - start < 5:
        start, end = len(time) // 2, min(len(time), len(time) // 2 + int(2.0 / dt))
    margin = max(1, int(round(0.25 / dt)))
    start = max(0, start - margin)
    end = min(len(time), end + margin)

    figure, axes = plt.subplots(4, 3, figsize=(16, 12), sharex="col")
    for row_index, (label, frame) in enumerate(data.items()):
        color = COLORS[label]
        ax = axes[row_index, 0]
        ax.plot(time, ref_pos, "k--", linewidth=1.4, label="POS reference")
        ax.plot(time, frame["act_pos"], color=color, linewidth=1.15, label="POS actual")
        ax.plot(time, ref_neg, color="0.45", linestyle="--", linewidth=1.4, label="NEG reference")
        ax.plot(time, frame["act_neg"], color=color, linestyle=":", linewidth=1.25, label="NEG actual")
        ax.set_ylabel(f"{label}\nPressure (kPa)")
        if row_index == 0:
            ax.set_title("Full trajectory")
            ax.legend(frameon=True, ncol=2, fontsize=8)

        ax = axes[row_index, 1]
        ax.plot(time[start:end], ref_pos[start:end], "k--", linewidth=1.5)
        ax.plot(time[start:end], frame["act_pos"].to_numpy()[start:end], color=color, linewidth=1.3)
        if row_index == 0:
            ax.set_title("Common plateau zoom: POS")

        ax = axes[row_index, 2]
        ax.plot(time[start:end], ref_neg[start:end], "k--", linewidth=1.5)
        ax.plot(time[start:end], frame["act_neg"].to_numpy()[start:end], color=color, linewidth=1.3)
        if row_index == 0:
            ax.set_title("Common plateau zoom: NEG")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    figure.suptitle(
        f"Representative held-out trajectory (member {member:02d}; dashed = reference)",
        fontsize=17,
        y=1.005,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_reward_effect_card(
    observation_label: str,
    base_label: str,
    smooth_label: str,
    model_root: Path,
    table: pd.DataFrame,
    output: Path,
    rolling_window: int,
) -> None:
    """Plot the reward comparison within one fixed observation family.

    This intentionally does not overlay all four arms.  The question is only
    whether the rate-mismatch objective changes a controller whose observation
    is already fixed (H or HR), which is the cleanest format for the meeting.
    """

    labels = (base_label, smooth_label)
    pair = table.set_index("arm").loc[list(labels)]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))

    axis = axes[0, 0]
    for label in labels:
        train = pd.read_csv(model_root / ARMS[label] / "train_diagnostics.csv")
        combined = np.sqrt((train["pos_rmse"] ** 2 + train["neg_rmse"] ** 2) / 2.0)
        axis.plot(
            train["epi"],
            rolling_mean(combined.to_numpy(), rolling_window),
            linewidth=2.0,
            color=COLORS[label],
            label=label,
        )
    axis.set_title("Training tracking error")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Combined RMSE (kPa)")
    axis.set_xlim(1, 2000)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=True)

    axis = axes[0, 1]
    values = pair["combined_rmse_mean_kpa"].to_numpy()
    errors = pair["combined_rmse_std_kpa"].to_numpy()
    bars = axis.bar(
        x,
        values,
        yerr=errors,
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axis.set_title("Fixed-bank tracking")
    axis.set_ylabel("Combined RMSE (kPa)")
    axis.set_xticks(x, labels)
    axis.set_ylim(bottom=0)

    axis = axes[1, 0]
    pos = pair["plateau_pos_rate_rms_mean_kpa_s"].to_numpy()
    neg = pair["plateau_neg_rate_rms_mean_kpa_s"].to_numpy()
    axis.bar(
        x - width / 2,
        pos,
        width,
        yerr=pair["plateau_pos_rate_rms_std_kpa_s"].to_numpy(),
        capsize=3,
        color="#4C78A8",
        label="POS",
    )
    axis.bar(
        x + width / 2,
        neg,
        width,
        yerr=pair["plateau_neg_rate_rms_std_kpa_s"].to_numpy(),
        capsize=3,
        color="#E45756",
        label="NEG",
    )
    axis.set_title("Pressure motion on reference plateaus")
    axis.set_ylabel(r"Pressure-rate RMS (kPa s$^{-1}$)")
    axis.set_xticks(x, labels)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=True)

    axis = axes[1, 1]
    values = pair["plateau_action_delta_rms_mean"].to_numpy()
    errors = pair["plateau_action_delta_rms_std"].to_numpy()
    bars = axis.bar(
        x,
        values,
        yerr=errors,
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axis.set_title("Applied-command variation on plateaus")
    axis.set_ylabel(r"RMS of $\Delta u$")
    axis.set_xticks(x, labels)
    axis.set_ylim(bottom=0)

    figure.suptitle(
        f"Reward-shaping check within {observation_label} observation\n"
        "seed 0; 2,000 episodes; final checkpoints; 16 fixed references",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_training_reward_by_observation(
    model_root: Path, output: Path, rolling_window: int
) -> None:
    """Show raw training-return evolution separately for H and HR.

    Base and smooth arms have different scalar objectives, so return is useful
    for convergence diagnostics only.  The plot keeps observation fixed within
    each panel; the full-bank RMSE/motion table remains the outcome metric.
    """

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    groups = (
        ("H: pressure/action history", ("H-base", "H-smooth")),
        ("HR: history + explicit pressure rate", ("HR-base", "HR-smooth")),
    )
    for axis, (title, labels) in zip(axes, groups):
        for label in labels:
            train = pd.read_csv(model_root / ARMS[label] / "train_diagnostics.csv")
            axis.plot(
                train["epi"],
                rolling_mean(train["reward"].to_numpy(), rolling_window),
                color=COLORS[label],
                linewidth=2.0,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Episode reward")
        axis.set_xlim(1, 2000)
        axis.set_ylim(top=0)
        axis.legend(frameon=True)
    figure.suptitle(
        f"Training reward ({rolling_window}-episode mean; convergence diagnostic)",
        fontsize=16,
        y=1.01,
    )
    figure.text(
        0.5,
        -0.02,
        "Base and smooth rewards are different objectives; do not compare their raw values as a common performance score.\n"
        "Use fixed-bank RMSE, plateau pressure-rate RMS, and command variation for the conclusion.",
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_reward_pair_tracking(
    observation_label: str,
    labels: tuple[str, str],
    evaluation_root: Path,
    member: int,
    output: Path,
) -> None:
    """Use the established POS-above-NEG tracking layout for a reward pair."""

    data = {
        label: pd.read_csv(evaluation_root / ARMS[label] / f"member_{member:02d}.csv")
        for label in labels
    }
    base = data[labels[0]]
    figure, axes = plt.subplots(2, 1, figsize=(12.4, 7.4), sharex=True)
    for axis, channel, reference, actual in (
        (axes[0], "POS", "ref_pos", "act_pos"),
        (axes[1], "NEG", "ref_neg", "act_neg"),
    ):
        axis.plot(
            base["time"],
            base[reference],
            "k--",
            linewidth=1.7,
            label="Reference",
        )
        for label in labels:
            axis.plot(
                data[label]["time"],
                data[label][actual],
                color=COLORS[label],
                linewidth=1.35,
                label=label,
            )
        axis.set_ylabel(f"{channel} pressure (kPa)")
        axis.grid(True, alpha=0.35)
        axis.legend(frameon=True, ncol=3, loc="best")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"Reward effect within {observation_label}: representative approximately-40-s tracking "
        f"(member {member:02d})",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_standard_three_signal_tracking(rollout: pd.DataFrame, output: Path) -> None:
    """Plot one policy in the common chamber/POS/NEG diagnostic format."""

    figure, axes = plt.subplots(3, 1, figsize=(10.2, 10.4), sharex=True)
    time = rollout["time"]

    axes[0].plot(time, rollout["chamber_pos"], color="red", linewidth=1.7, label="CH_POS")
    axes[0].plot(time, rollout["chamber_neg"], color="blue", linewidth=1.7, label="CH_NEG")
    axes[0].set_ylabel("Chamber [kPa]")
    axes[0].legend(frameon=True, loc="upper right")

    axes[1].plot(time, rollout["ref_pos"], color="black", linewidth=1.7, label="REF")
    axes[1].plot(time, rollout["act_pos"], color="red", linewidth=1.7, label="ACT_POS")
    axes[1].set_ylabel("Act Pos [kPa]")
    axes[1].legend(frameon=True, loc="upper right")

    axes[2].plot(time, rollout["ref_neg"], color="black", linewidth=1.7, label="REF")
    axes[2].plot(time, rollout["act_neg"], color="blue", linewidth=1.7, label="ACT_NEG")
    axes[2].set_ylabel("Act Neg [kPa]")
    axes[2].legend(frameon=True, loc="upper right")

    for axis in axes:
        axis.grid(True, which="major", color="0.70", linewidth=0.9)
        axis.minorticks_on()
        axis.grid(True, which="minor", color="0.82", linewidth=0.55)
        axis.set_xlim(float(time.iloc[0]), float(time.iloc[-1]))
        axis.set_xlabel("Time [sec]")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_effects(summaries: dict[str, dict], output: Path) -> None:
    comparisons = {
        "reward_effect_with_H": ("H-smooth", "H-base"),
        "reward_effect_with_HR": ("HR-smooth", "HR-base"),
        "rate_coordinate_effect_base": ("HR-base", "H-base"),
        "rate_coordinate_effect_smooth": ("HR-smooth", "H-smooth"),
    }
    keys = {
        "pos_rmse_kpa": "pos_rmse_kpa",
        "neg_rmse_kpa": "neg_rmse_kpa",
        "plateau_pos_rate_rms_kpa_s": "plateau_pos_pressure_rate_rms_kpa_s",
        "plateau_neg_rate_rms_kpa_s": "plateau_neg_pressure_rate_rms_kpa_s",
        "plateau_action_delta_rms": "plateau_action_delta_rms",
    }
    rows = []
    for comparison, (new_label, old_label) in comparisons.items():
        for metric, key in keys.items():
            new = metric_arrays(summaries[new_label], key)
            old = metric_arrays(summaries[old_label], key)
            difference, percent = paired_change(new, old)
            rows.append(
                {
                    "comparison": comparison,
                    "new_arm": new_label,
                    "old_arm": old_label,
                    "metric": metric,
                    "paired_mean_difference": difference,
                    "change_percent_of_member_means": percent,
                    "members_improved_fraction": float(np.mean(new < old)),
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("pneu_rl/models"))
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0831_histreward2x2_fullbank/seed0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0831_histreward2x2_analysis"),
    )
    parser.add_argument("--rolling-window", type=int, default=50)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_style()

    summaries = {
        label: load_summary(args.evaluation_root / model / "summary.yaml")
        for label, model in ARMS.items()
    }
    table = build_metrics_table(summaries)
    table.to_csv(args.output_root / "final_checkpoint_metrics.csv", index=False)
    write_effects(summaries, args.output_root / "paired_effects.csv")
    plot_summary(
        summaries,
        args.model_root,
        table,
        args.output_root / "history_rate_reward_2x2_summary.png",
        args.rolling_window,
    )
    member = choose_representative_member(summaries)
    plot_trajectory(
        args.evaluation_root,
        member,
        args.output_root / "history_rate_reward_2x2_trajectory.png",
    )
    plot_reward_effect_card(
        "H (pressure/action history only)",
        "H-base",
        "H-smooth",
        args.model_root,
        table,
        args.output_root / "history_only_reward_effect.png",
        args.rolling_window,
    )
    plot_reward_effect_card(
        "HR (history + explicit pressure rate)",
        "HR-base",
        "HR-smooth",
        args.model_root,
        table,
        args.output_root / "history_rate_reward_effect.png",
        args.rolling_window,
    )
    plot_training_reward_by_observation(
        args.model_root,
        args.output_root / "history_rate_reward_2x2_training_reward.png",
        args.rolling_window,
    )
    plot_reward_pair_tracking(
        "H (pressure/action history only)",
        ("H-base", "H-smooth"),
        args.evaluation_root,
        member,
        args.output_root / "history_only_reward_tracking.png",
    )
    plot_reward_pair_tracking(
        "HR (history + explicit pressure rate)",
        ("HR-base", "HR-smooth"),
        args.evaluation_root,
        member,
        args.output_root / "history_rate_reward_tracking.png",
    )
    for label, model in ARMS.items():
        rollout = pd.read_csv(args.evaluation_root / model / f"member_{member:02d}.csv")
        plot_standard_three_signal_tracking(
            rollout,
            args.output_root / f"{label.lower().replace('-', '_')}_standard_40s.png",
        )
    print(f"representative_member: {member:02d}")
    print(f"saved: {args.output_root}")


if __name__ == "__main__":
    main()
