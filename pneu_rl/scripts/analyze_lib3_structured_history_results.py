#!/usr/bin/env python3
"""Summarize the matched lib3 structured-history validation.

The evaluator writes one deterministic, 2,048-step approximately-40-s rollout for each
member of the fixed 16-trajectory reference bank.  This script keeps the
comparison paired by reference member and creates the figures needed for the
lab-meeting review:

* training and full-bank metric summary;
* matched-reward learning curves and convergence-time summary;
* a common representative approximately-40-s pressure-reference tracking trace;
* six physical valve-command traces for that same reference;
* tables of per-member metrics and paired changes relative to e_dot.

This is a seed-0 screening analysis.  Reference-member variation is shown as
an error bar, but it must not be interpreted as variation across training
seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ARMS = {
    "e_dot": "260906_lib3_edot_seed0",
    "PIDE": "260906_lib3_pide_seed0",
    "GPIDE-lite": "260906_lib3_gpide_seed0",
}

# Historical K=1 runs supplied the original rate/reserve context.  They have
# the same profile, horizon, action history, carry protocol, and seed as the
# 2026-09 structured screen, but chamber_bound_rwd_coeff differs (0.0 versus
# 0.1).  They are therefore plotted only as reward-learning context, not as a
# matched observation ablation.
REWARD_CONTEXT_ARMS = {
    "HR archive (rate)": "260810_lib3_ablation_profile_hist_rate_seed0",
    "HRR archive (rate + reserve)": "260810_lib3_ablation_profile_hist_rate_reserve_seed0",
    "e_dot screen": "260906_lib3_edot_seed0",
    "PIDE screen": "260906_lib3_pide_seed0",
    "GPIDE-lite screen": "260906_lib3_gpide_seed0",
}

COLORS = {
    "e_dot": "#4C78A8",
    "PIDE": "#E45756",
    "GPIDE-lite": "#54A24B",
}

REWARD_CONTEXT_COLORS = {
    "HR archive (rate)": "#8172B3",
    "HRR archive (rate + reserve)": "#F2A541",
    "e_dot screen": COLORS["e_dot"],
    "PIDE screen": COLORS["PIDE"],
    "GPIDE-lite screen": COLORS["GPIDE-lite"],
}

CONTROL_LABELS = (
    "CH POS",
    "CH NEG",
    "ACT POS IN",
    "ACT POS OUT",
    "ACT NEG IN",
    "ACT NEG OUT",
)

# These labels describe the *implemented* features, rather than claiming that
# every arm reproduces the full NeurIPS GPIDE architecture.
FEATURE_DESCRIPTIONS = {
    "e_dot": r"D-only tracking feature: $\dot e$",
    "PIDE": r"PID tracking feature: $e$, $\int e$, $\dot e$",
    "GPIDE-lite": r"Fixed difference/action feature: $\Delta o$, $u$, $\Delta u$",
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


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def members_frame(summary: dict, label: str) -> pd.DataFrame:
    frame = pd.DataFrame(summary["episodes"]).copy()
    if "reference_member" not in frame:
        frame["reference_member"] = np.arange(len(frame), dtype=int)
    frame["arm"] = label
    frame["combined_rmse_kpa"] = np.sqrt(
        (frame["pos_rmse_kpa"] ** 2 + frame["neg_rmse_kpa"] ** 2) / 2.0
    )
    frame["combined_plateau_rate_rms_kpa_s"] = np.sqrt(
        (
            frame["plateau_pos_pressure_rate_rms_kpa_s"] ** 2
            + frame["plateau_neg_pressure_rate_rms_kpa_s"] ** 2
        )
        / 2.0
    )
    frame["mean_tolerance_fraction"] = (
        frame["pos_tolerance_fraction"] + frame["neg_tolerance_fraction"]
    ) / 2.0
    frame["mean_direction_agreement"] = (
        frame["pos_direction_agreement"] + frame["neg_direction_agreement"]
    ) / 2.0
    frame["mean_reserve_ge_10_fraction"] = (
        frame["positive_reserve_ge_10kpa_fraction"]
        + frame["negative_reserve_ge_10kpa_fraction"]
    ) / 2.0
    return frame.sort_values("reference_member").reset_index(drop=True)


def aggregate_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    metrics = (
        "pos_rmse_kpa",
        "neg_rmse_kpa",
        "combined_rmse_kpa",
        "plateau_pos_pressure_rate_rms_kpa_s",
        "plateau_neg_pressure_rate_rms_kpa_s",
        "combined_plateau_rate_rms_kpa_s",
        "plateau_action_delta_rms",
        "physical_action_delta_abs_mean",
        "mean_tolerance_fraction",
        "mean_direction_agreement",
        "mean_reserve_ge_10_fraction",
    )
    for label, frame in frames.items():
        row = {"arm": label, "n_reference_members": int(len(frame))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(frames: dict[str, pd.DataFrame], baseline: str) -> pd.DataFrame:
    """Write paired fixed-reference changes; lower is better except fractions."""

    lower_is_better = (
        "pos_rmse_kpa",
        "neg_rmse_kpa",
        "combined_rmse_kpa",
        "plateau_pos_pressure_rate_rms_kpa_s",
        "plateau_neg_pressure_rate_rms_kpa_s",
        "combined_plateau_rate_rms_kpa_s",
        "plateau_action_delta_rms",
        "physical_action_delta_abs_mean",
    )
    higher_is_better = (
        "mean_tolerance_fraction",
        "mean_direction_agreement",
        "mean_reserve_ge_10_fraction",
    )
    base = frames[baseline].set_index("reference_member")
    rows = []
    for label, frame in frames.items():
        if label == baseline:
            continue
        candidate = frame.set_index("reference_member").reindex(base.index)
        for metric in lower_is_better + higher_is_better:
            delta = candidate[metric] - base[metric]
            if metric in lower_is_better:
                improved = delta < 0.0
            else:
                improved = delta > 0.0
            rows.append(
                {
                    "comparison": f"{label} vs {baseline}",
                    "metric": metric,
                    "mean_paired_change": float(delta.mean()),
                    "median_paired_change": float(delta.median()),
                    "relative_change_percent": float(
                        100.0 * (candidate[metric].mean() / base[metric].mean() - 1.0)
                    ),
                    "members_improved_fraction": float(improved.mean()),
                    "n_reference_members": int(len(delta)),
                }
            )
    return pd.DataFrame(rows)


def rolling_mean(values: pd.Series, window: int) -> np.ndarray:
    return values.rolling(window, min_periods=window).mean().to_numpy()


def first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0] + 1) if indices.size else None


def training_dynamics(
    model_root: Path, rolling_window: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return learning curves and predeclared convergence summaries.

    Reward is comparable only among these three arms because their reward
    configuration is identical.  ``reward_90pct_episode`` is the first
    episode at which the rolling reward reaches 90% of the arm's own gap from
    its initial rolling value to its final-100-episode mean.  This normalizes
    differing final values while retaining a reproducible, finite-budget
    measure of learning speed.
    """

    curves: dict[str, pd.DataFrame] = {}
    rows = []
    for label, model in ARMS.items():
        frame = pd.read_csv(model_root / model / "train_diagnostics.csv").copy()
        frame["combined_rmse_kpa"] = np.sqrt(
            (frame["pos_rmse"] ** 2 + frame["neg_rmse"] ** 2) / 2.0
        )
        frame["reward_rolling"] = rolling_mean(frame["reward"], rolling_window)
        frame["rmse_rolling"] = rolling_mean(
            frame["combined_rmse_kpa"], rolling_window
        )
        curves[label] = frame

        reward_initial = float(frame["reward_rolling"].iloc[rolling_window - 1])
        reward_final = float(frame["reward"].iloc[-100:].mean())
        reward_target = reward_initial + 0.9 * (reward_final - reward_initial)
        rows.append(
            {
                "arm": label,
                "rolling_window_episodes": int(rolling_window),
                "initial_reward_rolling": reward_initial,
                "final_reward_mean_last100": reward_final,
                "best_reward_rolling": float(frame["reward_rolling"].max()),
                "best_reward_rolling_episode": int(frame["reward_rolling"].idxmax() + 1),
                "reward_90pct_target": reward_target,
                "reward_90pct_episode": first_index(
                    frame["reward_rolling"].to_numpy() >= reward_target
                ),
                "mean_reward_episodes_1_200": float(frame["reward"].iloc[:200].mean()),
                "mean_reward_episodes_201_500": float(frame["reward"].iloc[200:500].mean()),
                "mean_reward_episodes_501_1000": float(frame["reward"].iloc[500:].mean()),
                "rmse_rolling_episode_le_2kpa": first_index(
                    frame["rmse_rolling"].to_numpy() <= 2.0
                ),
                "rmse_rolling_episode_le_1p5kpa": first_index(
                    frame["rmse_rolling"].to_numpy() <= 1.5
                ),
                "final_rmse_rolling": float(frame["rmse_rolling"].iloc[-1]),
                "best_rmse_rolling": float(frame["rmse_rolling"].min()),
                "best_rmse_rolling_episode": int(frame["rmse_rolling"].idxmin() + 1),
            }
        )
    return curves, pd.DataFrame(rows)


def plot_training_dynamics(
    curves: dict[str, pd.DataFrame], rolling_window: int, output: Path
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)
    for label, frame in curves.items():
        axes[0].plot(
            frame["epi"],
            frame["reward_rolling"],
            color=COLORS[label],
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            frame["epi"],
            frame["rmse_rolling"],
            color=COLORS[label],
            linewidth=2.0,
            label=label,
        )
    axes[0].set_title(f"Training return ({rolling_window}-episode mean; higher is better)")
    axes[0].set_ylabel("Episode reward")
    axes[0].legend(frameon=True, ncol=3)
    axes[0].set_ylim(top=0)
    axes[1].axhline(2.0, color="0.35", linestyle="--", linewidth=1.0, label="2 kPa")
    axes[1].axhline(1.5, color="0.55", linestyle=":", linewidth=1.0, label="1.5 kPa")
    axes[1].set_title("Training tracking error")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Combined RMSE (kPa)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=True, ncol=2)
    for axis in axes:
        axis.set_xlim(1, 1000)
    figure.suptitle(
        "Matched-reward learning dynamics (lib3, seed 0)", fontsize=16, y=1.01
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_reward_context_with_hr_hrr(
    model_root: Path, rolling_window: int, output: Path
) -> None:
    """Show the requested HR/HRR context with the PIDE-inspired screen.

    Raw reward is an optimization diagnostic, not a cross-arm performance
    metric here: the archive uses no chamber-bound reward whereas the newly
    matched e_dot/PIDE/GPIDE-lite screen uses a 0.1 coefficient.  Dashed
    archive curves make that boundary visible in the graphic itself.
    """

    figure, axis = plt.subplots(figsize=(11.8, 6.4))
    for label, model in REWARD_CONTEXT_ARMS.items():
        train = pd.read_csv(model_root / model / "train_diagnostics.csv")
        archive = label.startswith("HR")
        axis.plot(
            train["epi"],
            rolling_mean(train["reward"], rolling_window),
            color=REWARD_CONTEXT_COLORS[label],
            linestyle="--" if archive else "-",
            linewidth=2.0,
            label=label,
        )
    axis.set_title(
        f"Training reward context ({rolling_window}-episode mean; seed 0)"
    )
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode reward (higher is better within the same objective)")
    axis.set_xlim(1, 1000)
    axis.set_ylim(top=0)
    axis.legend(frameon=True, ncol=2, loc="lower right")
    axis.text(
        0.02,
        0.03,
        "solid: structured screen, chamber-bound coefficient = 0.1\n"
        "dashed: HR/HRR archive, chamber-bound coefficient = 0.0",
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.70", "alpha": 0.93},
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_summary(
    frames: dict[str, pd.DataFrame],
    model_root: Path,
    output: Path,
    rolling_window: int,
) -> None:
    labels = list(ARMS)
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))

    axis = axes[0, 0]
    for label, model in ARMS.items():
        train = pd.read_csv(model_root / model / "train_diagnostics.csv")
        combined = np.sqrt((train["pos_rmse"] ** 2 + train["neg_rmse"] ** 2) / 2.0)
        axis.plot(
            train["epi"],
            rolling_mean(combined, rolling_window),
            color=COLORS[label],
            linewidth=2.0,
            label=label,
        )
    axis.set_title(f"Training tracking error ({rolling_window}-episode mean)")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Combined RMSE (kPa)")
    axis.set_xlim(1, 1000)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=True)

    combined = [frames[label]["combined_rmse_kpa"] for label in labels]
    axis = axes[0, 1]
    axis.bar(
        x,
        [value.mean() for value in combined],
        yerr=[value.std(ddof=0) for value in combined],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    axis.set_title("Fixed-bank tracking")
    axis.set_ylabel("Combined RMSE (kPa)")
    axis.set_xticks(x, labels, rotation=14)

    axis = axes[0, 2]
    pos = [frames[label]["plateau_pos_pressure_rate_rms_kpa_s"] for label in labels]
    neg = [frames[label]["plateau_neg_pressure_rate_rms_kpa_s"] for label in labels]
    axis.bar(
        x - width / 2,
        [value.mean() for value in pos],
        width,
        yerr=[value.std(ddof=0) for value in pos],
        label="POS",
        color="#4C78A8",
        capsize=3,
    )
    axis.bar(
        x + width / 2,
        [value.mean() for value in neg],
        width,
        yerr=[value.std(ddof=0) for value in neg],
        label="NEG",
        color="#E45756",
        capsize=3,
    )
    axis.set_title("Pressure motion on reference plateaus")
    axis.set_ylabel(r"Pressure-rate RMS (kPa s$^{-1}$)")
    axis.set_xticks(x, labels, rotation=14)
    axis.legend(frameon=True)

    axis = axes[1, 0]
    values = [frames[label]["plateau_action_delta_rms"] for label in labels]
    axis.bar(
        x,
        [value.mean() for value in values],
        yerr=[value.std(ddof=0) for value in values],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    axis.set_title("Command variation on plateaus")
    axis.set_ylabel(r"RMS of $\Delta u$ (physical command)")
    axis.set_xticks(x, labels, rotation=14)

    axis = axes[1, 1]
    values = [100.0 * frames[label]["mean_reserve_ge_10_fraction"] for label in labels]
    axis.bar(
        x,
        [value.mean() for value in values],
        yerr=[value.std(ddof=0) for value in values],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    axis.set_title("Pneumatic reserve availability")
    axis.set_ylabel("Mean channels with reserve >= 10 kPa (%)")
    axis.set_ylim(0, 100.5)
    axis.set_xticks(x, labels, rotation=14)

    axis = axes[1, 2]
    for label, frame in frames.items():
        axis.scatter(
            frame["combined_rmse_kpa"].mean(),
            frame["combined_plateau_rate_rms_kpa_s"].mean(),
            s=100,
            color=COLORS[label],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        axis.annotate(
            label,
            (
                frame["combined_rmse_kpa"].mean(),
                frame["combined_plateau_rate_rms_kpa_s"].mean(),
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set_title("Tracking-motion trade-off (lower is better)")
    axis.set_xlabel("Combined RMSE (kPa)")
    axis.set_ylabel(r"Combined plateau-rate RMS (kPa s$^{-1}$)")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)

    figure.suptitle(
        "lib3 structured-history screen (seed 0; 16 fixed references; 2,048 steps/member)",
        fontsize=17,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def choose_representative_member(frames: dict[str, pd.DataFrame]) -> int:
    aligned = pd.concat(
        [
            frame.set_index("reference_member")["combined_rmse_kpa"].rename(label)
            for label, frame in frames.items()
        ],
        axis=1,
    )
    mean_by_member = aligned.mean(axis=1)
    median = float(mean_by_member.median())
    return int((mean_by_member - median).abs().idxmin())


def plot_tracking(
    data: dict[str, pd.DataFrame], member: int, output: Path
) -> None:
    base = next(iter(data.values()))
    time = base["time"].to_numpy()
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True)
    channels = (("POS", "ref_pos", "act_pos"), ("NEG", "ref_neg", "act_neg"))
    for axis, (name, reference, actual) in zip(axes, channels):
        axis.plot(time, base[reference], "k--", linewidth=1.8, label="Reference")
        for label, frame in data.items():
            axis.plot(
                time,
                frame[actual],
                linewidth=1.45,
                color=COLORS[label],
                label=label,
            )
        axis.set_ylabel(f"{name} pressure (kPa)")
        axis.grid(True, alpha=0.35)
        axis.legend(frameon=True, ncol=4, loc="best")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"Common approximately-40-s reference tracking (bank member {member:02d})",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_commands(data: dict[str, pd.DataFrame], member: int, output: Path) -> None:
    base = next(iter(data.values()))
    time = base["time"].to_numpy()
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex=True)
    for index, axis in enumerate(axes.flat, start=1):
        for label, frame in data.items():
            axis.plot(
                time,
                frame[f"ctrl{index}"],
                linewidth=1.15,
                color=COLORS[label],
                label=label,
            )
        axis.set_title(CONTROL_LABELS[index - 1])
        axis.set_ylabel("Physical command")
        axis.set_ylim(0.79, 1.01)
        axis.grid(True, alpha=0.35)
        if index <= 2:
            axis.legend(frameon=True, ncol=3, loc="upper right")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        f"Six-valve commands on the common approximately-40-s reference (member {member:02d})",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_single_arm_card(
    label: str,
    curve: pd.DataFrame,
    frame: pd.DataFrame,
    rollout: pd.DataFrame,
    member: int,
    output: Path,
    rolling_window: int,
) -> None:
    """Make one self-contained result card per structured feature.

    The previous six-panel comparison is useful for an overview, but it makes
    it hard to discuss each feature as a separate experimental question during
    a meeting.  A card keeps the learning curve, reference-bank distribution,
    and a common 40-s tracking trace for exactly one feature together.
    """

    figure = plt.figure(figsize=(13.2, 11.4))
    grid = figure.add_gridspec(3, 2, height_ratios=(0.78, 1.0, 1.0))
    axes = {
        "training": figure.add_subplot(grid[0, 0]),
        "distribution": figure.add_subplot(grid[0, 1]),
        "pos": figure.add_subplot(grid[1, :]),
        "neg": figure.add_subplot(grid[2, :]),
    }
    color = COLORS[label]

    axis = axes["training"]
    axis.plot(
        curve["epi"],
        curve["rmse_rolling"],
        color=color,
        linewidth=2.0,
        label=f"{rolling_window}-episode mean",
    )
    axis.axhline(2.0, color="0.35", linestyle="--", linewidth=1.0, label="2 kPa")
    axis.axhline(1.5, color="0.55", linestyle=":", linewidth=1.0, label="1.5 kPa")
    axis.set_title("Training tracking error")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Combined RMSE (kPa)")
    axis.set_xlim(1, int(curve["epi"].max()))
    axis.set_ylim(bottom=0)
    axis.legend(frameon=True)

    axis = axes["distribution"]
    member_index = frame["reference_member"].to_numpy()
    values = frame["combined_rmse_kpa"].to_numpy()
    axis.plot(member_index, values, color=color, linewidth=1.2, alpha=0.75)
    axis.scatter(member_index, values, color=color, edgecolor="black", linewidth=0.45, zorder=3)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    axis.axhline(mean, color="0.15", linestyle="--", linewidth=1.1)
    axis.text(
        0.03,
        0.95,
        f"mean = {mean:.3f} ± {std:.3f} kPa\n"
        f"plateau rate RMS = {frame['combined_plateau_rate_rms_kpa_s'].mean():.2f} kPa/s\n"
        f"plateau action Δ RMS = {frame['plateau_action_delta_rms'].mean():.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    axis.set_title("Fixed-bank tracking distribution")
    axis.set_xlabel("Reference-bank member")
    axis.set_ylabel("Combined RMSE (kPa)")
    axis.set_xlim(member_index.min() - 0.5, member_index.max() + 0.5)
    axis.set_ylim(bottom=0)

    # Keep the same vertical POS-then-NEG convention as the established
    # approximately-40-s tracking figures.  It is much easier to read on a
    # slide than placing the two physical chambers side by side.
    for axis, channel, reference, actual in (
        (axes["pos"], "POS", "ref_pos", "act_pos"),
        (axes["neg"], "NEG", "ref_neg", "act_neg"),
    ):
        axis.plot(
            rollout["time"],
            rollout[reference],
            "k--",
            linewidth=1.5,
            label="Reference",
        )
        axis.plot(
            rollout["time"],
            rollout[actual],
            color=color,
            linewidth=1.25,
            label=label,
        )
        axis.set_title(f"Representative 40-s tracking: {channel}")
        if channel == "NEG":
            axis.set_xlabel("Time (s)")
        axis.set_ylabel("Pressure (kPa)")
        axis.grid(True, alpha=0.35)
        axis.legend(frameon=True)

    figure.suptitle(
        f"lib3 structured-feature screen — {label}\n"
        f"{FEATURE_DESCRIPTIONS[label]}  |  seed 0, 16 fixed references, member {member:02d}",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_single_arm_commands(
    label: str,
    rollout: pd.DataFrame,
    member: int,
    output: Path,
) -> None:
    """Plot the actual six valve commands for one feature, not overlaid arms."""

    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), sharex=True)
    for index, axis in enumerate(axes.flat, start=1):
        axis.plot(
            rollout["time"],
            rollout[f"ctrl{index}"],
            color=COLORS[label],
            linewidth=1.0,
        )
        axis.set_title(CONTROL_LABELS[index - 1])
        axis.set_ylabel("Physical command")
        axis.set_ylim(0.79, 1.01)
        axis.grid(True, alpha=0.35)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        f"Six applied valve commands — {label} (representative member {member:02d})",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_standard_three_signal_tracking(
    rollout: pd.DataFrame,
    output: Path,
) -> None:
    """Create the established chamber/POS/NEG diagnostic for one policy.

    This is deliberately controller-agnostic: red always means the positive
    chamber/actuator signal, blue always means the negative signal, and the
    black line is always the actuator reference.  Thus every result inserted
    in the markdown is visually comparable without decoding policy colors.
    """

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
    for axis in axes:
        axis.set_xlabel("Time [sec]")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_readme(
    output: Path,
    member: int,
    table: pd.DataFrame,
) -> None:
    compact = table[
        [
            "arm",
            "combined_rmse_kpa_mean",
            "combined_plateau_rate_rms_kpa_s_mean",
            "plateau_action_delta_rms_mean",
            "mean_reserve_ge_10_fraction_mean",
        ]
    ].copy()
    compact["mean_reserve_ge_10_fraction_mean"] *= 100.0
    # Avoid pandas.DataFrame.to_markdown(): the training environment does not
    # install its optional ``tabulate`` dependency.
    headings = list(compact.columns)
    table_lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---"] * len(headings)) + " |",
    ]
    for _, row in compact.iterrows():
        values = [
            str(row["arm"]),
            *[f"{float(row[column]):.3f}" for column in headings[1:]],
        ]
        table_lines.append("| " + " | ".join(values) + " |")
    table_text = "\n".join(table_lines)
    output.write_text(
        "# lib3 structured-history validation (seed 0)\n\n"
        "- **Protocol:** deterministic final checkpoint; 16 fixed members of the "
        "training reference bank; 2,048 policy steps/member at 50 Hz (nominal "
        "40.96 s).  The logged time ends at 40.86 s because the first 0.10 s "
        "of the delay warm-up repeats the initial timestamp.\n"
        "- **Fair comparison:** `e_dot`, `PIDE`, and `GPIDE-lite` share the "
        "same reward, reference profile, history length, carry setting, SAC "
        "budget, and seed. Only their appended structured observation feature "
        "differs.\n"
        "- **Uncertainty boundary:** error bars quantify variation over reference "
        "members, **not** training seeds or a held-out task distribution.\n"
        "- **Representative trace:** member "
        f"`{member:02d}` is selected as the median difficulty using the mean "
        "combined RMSE across the three controllers.\n\n"
        "## Full-bank summary\n\n"
        f"{table_text}\n\n"
        "`mean_reserve_ge_10_fraction_mean` is reported in percent in this compact "
        "table. Lower is better for RMSE, plateau pressure-rate RMS, and command "
        "delta RMS; higher reserve availability is better.\n\n"
        "## Files\n\n"
        "- `structured_history_summary.png`: training and full-bank summary.\n"
        "- `training_reward_rmse.png`: matched-reward training return and RMSE curves.\n"
        "- `learning_dynamics.csv`: reproducible convergence-time summary.\n"
        "- `representative_40s_tracking.png`: common approximately-40-s pressure/reference trace.\n"
        "- `representative_40s_commands.png`: all six physical valve commands.\n"
        "- `e_dot_feature_card.png`, `pide_feature_card.png`, and "
        "`gpide_lite_feature_card.png`: one self-contained learning/validation "
        "card per feature.\n"
        "- `e_dot_commands_40s.png`, `pide_commands_40s.png`, and "
        "`gpide_lite_commands_40s.png`: the applied six-valve commands for "
        "each feature separately.\n"
        "- `e_dot_standard_40s.png`, `pide_standard_40s.png`, and "
        "`gpide_lite_standard_40s.png`: common three-row chamber/POS/NEG "
        "diagnostics used in the markdown report.\n"
        "- `per_member_metrics.csv`: the paired rows used for all summaries.\n"
        "- `paired_effects_vs_edot.csv`: per-reference paired changes relative to "
        "the minimal `e_dot` feature.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0906_structured_history_validation_actual2048"),
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path("pneu_rl/models")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0906_structured_history_analysis"),
    )
    parser.add_argument("--rolling-window", type=int, default=50)
    args = parser.parse_args()

    missing = [
        str(args.evaluation_root / model / "summary.yaml")
        for model in ARMS.values()
        if not (args.evaluation_root / model / "summary.yaml").is_file()
    ]
    if missing:
        raise FileNotFoundError("validation incomplete:\n" + "\n".join(missing))
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_style()

    summaries = {
        label: load_yaml(args.evaluation_root / model / "summary.yaml")
        for label, model in ARMS.items()
    }
    frames = {label: members_frame(summary, label) for label, summary in summaries.items()}
    per_member = pd.concat(frames.values(), ignore_index=True)
    per_member.to_csv(args.output_root / "per_member_metrics.csv", index=False)
    table = aggregate_table(frames)
    table.to_csv(args.output_root / "fullbank_metrics.csv", index=False)
    curves, dynamics = training_dynamics(args.model_root, args.rolling_window)
    dynamics.to_csv(args.output_root / "learning_dynamics.csv", index=False)
    plot_training_dynamics(
        curves,
        args.rolling_window,
        args.output_root / "training_reward_rmse.png",
    )
    plot_reward_context_with_hr_hrr(
        args.model_root,
        args.rolling_window,
        args.output_root / "feature_screen_with_hr_hrr_training_reward.png",
    )
    paired_effects(frames, baseline="e_dot").to_csv(
        args.output_root / "paired_effects_vs_edot.csv", index=False
    )
    plot_summary(
        frames,
        args.model_root,
        args.output_root / "structured_history_summary.png",
        args.rolling_window,
    )
    member = choose_representative_member(frames)
    data = {
        label: pd.read_csv(
            args.evaluation_root / model / f"member_{member:02d}.csv"
        )
        for label, model in ARMS.items()
    }
    plot_tracking(data, member, args.output_root / "representative_40s_tracking.png")
    plot_commands(data, member, args.output_root / "representative_40s_commands.png")
    for label in ARMS:
        slug = label.lower().replace("-", "_")
        plot_single_arm_card(
            label,
            curves[label],
            frames[label],
            data[label],
            member,
            args.output_root / f"{slug}_feature_card.png",
            args.rolling_window,
        )
        plot_single_arm_commands(
            label,
            data[label],
            member,
            args.output_root / f"{slug}_commands_40s.png",
        )
        plot_standard_three_signal_tracking(
            data[label],
            args.output_root / f"{slug}_standard_40s.png",
        )
    write_readme(args.output_root / "README.md", member, table)
    print(f"representative_member: {member:02d}")
    print(f"saved: {args.output_root}")


if __name__ == "__main__":
    main()
