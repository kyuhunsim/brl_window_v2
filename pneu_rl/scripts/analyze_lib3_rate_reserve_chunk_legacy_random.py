#!/usr/bin/env python3
"""Aggregate the lib3 rate/reserve/chunk controllers on legacy RandomRef.

The four RandomRef seeds are intentionally reported separately from the fixed
capacity-reference bank.  They inject discontinuous offset/parameter changes,
so they are an out-of-distribution jump-robustness screen rather than a
replacement for nominal tracking validation.
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
        return yaml.safe_load(handle)


def command_tv_per_sec(frame: pd.DataFrame) -> float:
    commands = frame[[f"ctrl{index}" for index in range(1, 7)]].to_numpy()
    duration = max(float(frame["time"].iloc[-1] - frame["time"].iloc[0]), 1e-9)
    return float(np.abs(np.diff(commands, axis=0)).sum() / (duration * 6.0))


def read_seed(root: Path, reference_seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for label, model in ARMS.items():
        summary_path = root / model / "summary.yaml"
        csv_path = root / model / "episode_01.csv"
        if not summary_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"missing completed rollout: {root / model}")
        episode = load_yaml(summary_path)["episodes"][0]
        row = dict(episode)
        row["arm"] = label
        row["reference_seed"] = reference_seed
        row["combined_rmse_kpa"] = float(
            np.hypot(row["pos_rmse_kpa"], row["neg_rmse_kpa"]) / np.sqrt(2.0)
        )
        row["mean_tolerance_fraction"] = float(
            (row["pos_tolerance_fraction"] + row["neg_tolerance_fraction"]) / 2.0
        )
        row["command_tv_per_channel_sec"] = command_tv_per_sec(pd.read_csv(csv_path))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ARMS:
        data = frame.loc[frame["arm"] == label]
        rows.append(
            {
                "arm": label,
                "n_reference_seeds": int(len(data)),
                "combined_rmse_mean_kpa": float(data["combined_rmse_kpa"].mean()),
                "combined_rmse_std_kpa": float(data["combined_rmse_kpa"].std(ddof=0)),
                "pos_rmse_mean_kpa": float(data["pos_rmse_kpa"].mean()),
                "neg_rmse_mean_kpa": float(data["neg_rmse_kpa"].mean()),
                "mean_tolerance_fraction": float(data["mean_tolerance_fraction"].mean()),
                "plateau_pos_rate_rms_mean_kpa_s": float(
                    data["plateau_pos_pressure_rate_rms_kpa_s"].mean()
                ),
                "plateau_neg_rate_rms_mean_kpa_s": float(
                    data["plateau_neg_pressure_rate_rms_kpa_s"].mean()
                ),
                "command_tv_per_channel_sec_mean": float(
                    data["command_tv_per_channel_sec"].mean()
                ),
                "command_tv_per_channel_sec_std": float(
                    data["command_tv_per_channel_sec"].std(ddof=0)
                ),
                "reference_max_abs_rate_mean_kpa_s": float(
                    data[["ref_pos_rate_abs_max_kpa_s", "ref_neg_rate_abs_max_kpa_s"]]
                    .to_numpy()
                    .max(axis=1)
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(table: pd.DataFrame, output: Path) -> None:
    labels = list(ARMS)
    x = np.arange(len(labels), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    axes[0].bar(
        x,
        table["combined_rmse_mean_kpa"],
        yerr=table["combined_rmse_std_kpa"],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    axes[0].set_title("Jump-reference tracking")
    axes[0].set_ylabel("Combined RMSE (kPa); lower is better")

    width = 0.34
    axes[1].bar(
        x - width / 2,
        table["plateau_pos_rate_rms_mean_kpa_s"],
        width,
        color="#4C78A8",
        label="POS pressure rate",
    )
    axes[1].bar(
        x + width / 2,
        table["plateau_neg_rate_rms_mean_kpa_s"],
        width,
        color="#E45756",
        label="NEG pressure rate",
    )
    axes[1].set_title("Pressure motion on reference plateaus")
    axes[1].set_ylabel(r"Rate RMS (kPa s$^{-1}$)")
    axes[1].legend(frameon=True)

    axes[2].bar(
        x,
        table["command_tv_per_channel_sec_mean"],
        yerr=table["command_tv_per_channel_sec_std"],
        color=[COLORS[label] for label in labels],
        capsize=3,
    )
    axes[2].set_title("Physical command bandwidth")
    axes[2].set_ylabel("Mean channel command TV (s$^{-1}$)")

    for axis in axes:
        axis.set_xticks(x, labels, rotation=14)
    figure.suptitle(
        "lib3 legacy RandomRef jump-robustness screen (4 reference seeds; one ~40 s rollout each)",
        fontsize=15,
        y=1.03,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[61098, 61099, 61100, 61101])
    parser.add_argument(
        "--root-template",
        type=str,
        default="pneu_rl/exp/lib3_0907_rate_reserve_chunk_legacy_random_seed{seed}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_0907_rate_reserve_chunk_legacy_random_analysis"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_style()
    per_seed = pd.concat(
        [read_seed(Path(args.root_template.format(seed=seed)), seed) for seed in args.seeds],
        ignore_index=True,
    )
    table = aggregate(per_seed)
    per_seed.to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    table.to_csv(args.output_root / "legacy_random_metrics.csv", index=False)
    plot_summary(table, args.output_root / "legacy_random_summary.png")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved: {args.output_root}")


if __name__ == "__main__":
    main()
