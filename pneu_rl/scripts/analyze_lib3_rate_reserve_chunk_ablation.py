#!/usr/bin/env python3
"""Analyze the historical lib3 rate/reserve/action-chunk ablation fairly.

The K=2 policy emits two open-loop physical commands per policy decision.
Consequently this script uses physical simulator steps on learning-curve x
axes and command total variation per physical second in the final comparison.
It deliberately does not invent the missing H+rate, K=2 control arm.
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
    "H+rate (K=1)": "260810_lib3_ablation_profile_hist_rate_seed0",
    "H+rate+reserve (K=1)": "260810_lib3_ablation_profile_hist_rate_reserve_seed0",
    "H+rate+reserve (K=2)": "260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0",
}

COLORS = {
    "H+rate (K=1)": "#4C78A8",
    "H+rate+reserve (K=1)": "#E45756",
    "H+rate+reserve (K=2)": "#54A24B",
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
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.full_load(handle)


def rolling(values: pd.Series, window: int) -> np.ndarray:
    return values.rolling(window, min_periods=window).mean().to_numpy()


def chunk_reward_weight(chunk_steps: int, gamma: float) -> float:
    if abs(gamma - 1.0) < 1e-12:
        return float(chunk_steps)
    return float((1.0 - gamma**chunk_steps) / (1.0 - gamma))


def command_tv_per_sec(frame: pd.DataFrame) -> float:
    control = frame[[f"ctrl{index}" for index in range(1, 7)]].to_numpy()
    time = frame["time"].to_numpy()
    duration = max(float(time[-1] - time[0]), 1e-9)
    # Mean channel-wise variation rate; unlike mean |Delta u| it remains
    # comparable when K=2 records half as many policy decisions as K=1.
    return float(np.abs(np.diff(control, axis=0)).sum() / (duration * 6.0))


def read_arm(
    label: str,
    model: str,
    model_root: Path,
    evaluation_root: Path,
    rolling_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = load_yaml(model_root / model / "cfg.yaml")
    summary = load_yaml(evaluation_root / model / "summary.yaml")
    env = cfg["env"]
    model_cfg = cfg["model"]
    chunk = int(env["action_chunk_steps"])
    horizon = int(model_cfg["horizon"])
    gamma = float(env["action_chunk_gamma"])
    physical_steps_per_episode = horizon * chunk

    train = pd.read_csv(model_root / model / "train_diagnostics.csv").copy()
    train["combined_rmse_kpa"] = np.sqrt(
        (train["pos_rmse"] ** 2 + train["neg_rmse"] ** 2) / 2.0
    )
    train["physical_steps"] = train["epi"] * physical_steps_per_episode
    # K=2 reports a discounted sum over two physical rewards at each policy
    # decision.  Normalize by the matching discounted physical-step count.
    train["reward_per_physical_step"] = train["reward"] / (
        horizon * chunk_reward_weight(chunk, gamma)
    )
    train["reward_rolling"] = rolling(
        train["reward_per_physical_step"], rolling_window
    )
    train["rmse_rolling"] = rolling(train["combined_rmse_kpa"], rolling_window)

    per_member = pd.DataFrame(summary["episodes"]).copy()
    per_member["combined_rmse_kpa"] = np.sqrt(
        (per_member["pos_rmse_kpa"] ** 2 + per_member["neg_rmse_kpa"] ** 2) / 2.0
    )
    per_member["arm"] = label
    per_member["action_chunk_steps"] = chunk
    tv_values = []
    for member in per_member["reference_member"]:
        csv_path = evaluation_root / model / f"member_{int(member):02d}.csv"
        tv_values.append(command_tv_per_sec(pd.read_csv(csv_path)))
    per_member["command_tv_per_channel_sec"] = tv_values
    meta = {
        "model": model,
        "action_chunk_steps": chunk,
        "horizon_policy_steps": horizon,
        "physical_steps_per_episode": physical_steps_per_episode,
        "reward_weighted_physical_steps_per_episode": horizon
        * chunk_reward_weight(chunk, gamma),
    }
    return train, per_member, meta


def aggregate(per_member: dict[str, pd.DataFrame], meta: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, frame in per_member.items():
        rows.append(
            {
                "arm": label,
                **meta[label],
                "n_reference_members": int(len(frame)),
                "combined_rmse_mean_kpa": float(frame["combined_rmse_kpa"].mean()),
                "combined_rmse_std_kpa": float(frame["combined_rmse_kpa"].std(ddof=0)),
                "pos_rmse_mean_kpa": float(frame["pos_rmse_kpa"].mean()),
                "neg_rmse_mean_kpa": float(frame["neg_rmse_kpa"].mean()),
                "plateau_pos_rate_rms_mean_kpa_s": float(
                    frame["plateau_pos_pressure_rate_rms_kpa_s"].mean()
                ),
                "plateau_neg_rate_rms_mean_kpa_s": float(
                    frame["plateau_neg_pressure_rate_rms_kpa_s"].mean()
                ),
                "command_tv_per_channel_sec_mean": float(
                    frame["command_tv_per_channel_sec"].mean()
                ),
                "command_tv_per_channel_sec_std": float(
                    frame["command_tv_per_channel_sec"].std(ddof=0)
                ),
                "mean_tolerance_fraction": float(
                    (
                        frame["pos_tolerance_fraction"]
                        + frame["neg_tolerance_fraction"]
                    ).mean()
                    / 2.0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(
    train: dict[str, pd.DataFrame],
    per_member: dict[str, pd.DataFrame],
    table: pd.DataFrame,
    output: Path,
) -> None:
    labels = list(ARMS)
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.4))
    for label, frame in train.items():
        axes[0, 0].plot(
            frame["physical_steps"],
            frame["reward_rolling"],
            color=COLORS[label],
            linewidth=2.0,
            label=label,
        )
        axes[0, 1].plot(
            frame["physical_steps"],
            frame["rmse_rolling"],
            color=COLORS[label],
            linewidth=2.0,
            label=label,
        )
    axes[0, 0].set_title("Training reward per physical step (50-episode mean)")
    axes[0, 0].set_ylabel("Normalized reward")
    axes[0, 1].set_title("Training tracking error (50-episode mean)")
    axes[0, 1].set_ylabel("Combined RMSE (kPa)")
    for axis in axes[0, :]:
        axis.set_xlabel("Physical simulator steps consumed")
        axis.set_xlim(left=0)
        axis.legend(frameon=True)

    x = np.arange(len(labels), dtype=float)
    values = [per_member[label]["combined_rmse_kpa"] for label in labels]
    axes[1, 0].bar(
        x,
        [value.mean() for value in values],
        yerr=[value.std(ddof=0) for value in values],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    axes[1, 0].set_title("Fixed-bank tracking (lower is better)")
    axes[1, 0].set_ylabel("Combined RMSE (kPa)")
    axes[1, 0].set_xticks(x, labels, rotation=14)

    width = 0.34
    pos = table["plateau_pos_rate_rms_mean_kpa_s"].to_numpy()
    neg = table["plateau_neg_rate_rms_mean_kpa_s"].to_numpy()
    axes[1, 1].bar(x - width / 2, pos, width, color="#4C78A8", label="POS rate")
    axes[1, 1].bar(x + width / 2, neg, width, color="#E45756", label="NEG rate")
    for index, row in table.iterrows():
        axes[1, 1].annotate(
            f"TV={row['command_tv_per_channel_sec_mean']:.3f}/s",
            (x[index], max(pos[index], neg[index])),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[1, 1].set_title("Plateau pressure motion; annotation = command TV/s")
    axes[1, 1].set_ylabel(r"Pressure-rate RMS (kPa s$^{-1}$)")
    axes[1, 1].set_xticks(x, labels, rotation=14)
    axes[1, 1].legend(frameon=True)

    figure.suptitle(
        "lib3 rate/reserve/action-chunk archive (seed 0; 16 fixed references)",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def representative_member(per_member: dict[str, pd.DataFrame]) -> int:
    aligned = pd.concat(
        [
            frame.set_index("reference_member")["combined_rmse_kpa"].rename(label)
            for label, frame in per_member.items()
        ],
        axis=1,
    )
    means = aligned.mean(axis=1)
    return int((means - means.median()).abs().idxmin())


def plot_tracking(
    evaluation_root: Path, member: int, output: Path
) -> None:
    data = {
        label: pd.read_csv(evaluation_root / model / f"member_{member:02d}.csv")
        for label, model in ARMS.items()
    }
    base = next(iter(data.values()))
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.7), sharex=True)
    for axis, ref_column, act_column, title in (
        (axes[0], "ref_pos", "act_pos", "POS"),
        (axes[1], "ref_neg", "act_neg", "NEG"),
    ):
        axis.plot(base["time"], base[ref_column], "k--", linewidth=1.7, label="Reference")
        for label, frame in data.items():
            axis.plot(
                frame["time"],
                frame[act_column],
                color=COLORS[label],
                linewidth=1.35,
                label=label,
            )
        axis.set_ylabel(f"{title} pressure (kPa)")
        axis.legend(frameon=True, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"Common approximately-40-s tracking trace (reference member {member:02d})",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_commands(evaluation_root: Path, member: int, output: Path) -> None:
    """Show every applied physical command, including both K=2 substeps."""

    data = {
        label: pd.read_csv(evaluation_root / model / f"member_{member:02d}.csv")
        for label, model in ARMS.items()
    }
    names = (
        "CH POS",
        "CH NEG",
        "ACT POS IN",
        "ACT POS OUT",
        "ACT NEG IN",
        "ACT NEG OUT",
    )
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex=True)
    for index, axis in enumerate(axes.flat, start=1):
        for label, frame in data.items():
            axis.plot(
                frame["time"],
                frame[f"ctrl{index}"],
                color=COLORS[label],
                linewidth=1.0,
                label=label,
            )
        axis.set_title(names[index - 1])
        axis.set_ylabel("Physical command")
        axis.set_ylim(0.79, 1.01)
        if index <= 2:
            axis.legend(frameon=True, ncol=3, fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        f"Applied six-valve commands (reference member {member:02d})", fontsize=16
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("pneu_rl/models"))
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0907_rate_reserve_chunk_validation_actual2048"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0907_rate_reserve_chunk_analysis"),
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
    train: dict[str, pd.DataFrame] = {}
    per_member: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    for label, model in ARMS.items():
        train[label], per_member[label], metadata[label] = read_arm(
            label,
            model,
            args.model_root,
            args.evaluation_root,
            args.rolling_window,
        )
    table = aggregate(per_member, metadata)
    table.to_csv(args.output_root / "fullbank_metrics.csv", index=False)
    pd.concat(per_member.values(), ignore_index=True).to_csv(
        args.output_root / "per_member_metrics.csv", index=False
    )
    plot_summary(train, per_member, table, args.output_root / "rate_reserve_chunk_summary.png")
    member = representative_member(per_member)
    plot_tracking(
        args.evaluation_root,
        member,
        args.output_root / "representative_40s_tracking.png",
    )
    plot_commands(
        args.evaluation_root,
        member,
        args.output_root / "representative_40s_commands.png",
    )
    print(f"representative_member: {member:02d}")
    print(f"saved: {args.output_root}")


if __name__ == "__main__":
    main()
