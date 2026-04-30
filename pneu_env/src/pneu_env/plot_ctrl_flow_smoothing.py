#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from utils.utils import setup_plot_style


CTRL_COLS = [f"ctrl{i}" for i in range(1, 7)]
FLOW_COLS = [f"flow{i}" for i in range(1, 7)]

setup_plot_style({"legend.fontsize": 18})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw and smoothed ctrl1~6 / flow1~6 signals from a CSV."
    )
    parser.add_argument("csv", help="Input CSV path")
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="uniform_filter1d window size. 1 means no smoothing.",
    )
    parser.add_argument("--start", type=float, default=None, help="Start time filter")
    parser.add_argument("--end", type=float, default=None, help="End time filter")
    parser.add_argument(
        "--no-rebase",
        action="store_true",
        help="Plot absolute CSV time instead of rebasing the selected range to 0 sec.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Default: CSV_stem_smooth_w{window_size}",
    )
    parser.add_argument("--no-show", action="store_true", help="Save only; do not show plot windows")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def smooth(values: np.ndarray, window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window_size <= 1:
        return arr.copy()
    return uniform_filter1d(arr, size=int(window_size))


def time_col(df: pd.DataFrame, path: Path) -> str:
    if "time" in df.columns:
        return "time"
    if "curr_time" in df.columns:
        return "curr_time"
    raise ValueError(f"{path}: missing time/curr_time column")


def require_cols(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}. Got: {list(df.columns)}")


def load_csv(path: Path, start: float | None, end: float | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    t_col = time_col(df, path)
    require_cols(df, CTRL_COLS + FLOW_COLS, path)

    df = df.sort_values(t_col).reset_index(drop=True)
    if start is not None:
        df = df[df[t_col] >= float(start)]
    if end is not None:
        df = df[df[t_col] <= float(end)]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows after time filtering.")
    return df, t_col


def plot_group(
    t: np.ndarray,
    df: pd.DataFrame,
    cols: list[str],
    *,
    window_size: int,
    title: str,
    ylabel: str,
    start: float | None,
    end: float | None,
    rebased: bool,
) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(21.0, 14.4),
        sharex=False,
        facecolor="w",
        constrained_layout=True,
    )
    axes = axes.reshape(-1)
    time_note = "rebased time" if rebased else "absolute time"
    range_note = f"start={start if start is not None else 'begin'}, end={end if end is not None else 'end'}"
    fig.suptitle(
        f"{title} | raw vs smooth | window={window_size} | {range_note} | {time_note}",
        fontsize=15,
    )

    for ax, col in zip(axes, cols):
        raw = df[col].to_numpy(dtype=np.float64)
        smoothed = smooth(raw, window_size)
        mae = float(np.mean(np.abs(raw - smoothed)))

        ax.plot(t, raw, "k-", linewidth=2.2, alpha=0.78, label="raw")
        ax.plot(t, smoothed, "r--", linewidth=2.2, alpha=0.95, label="smooth")
        ax.set_title(f"{col} | mean |diff|={mae:.4g}", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Time [sec]", fontsize=12)
        ax.tick_params(axis="both", labelsize=11, labelbottom=True)
        ax.grid(True, alpha=0.32)
        ax.legend(loc="upper right", fontsize=10)

    return fig


def main() -> None:
    args = parse_args()
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")

    csv_path = Path(args.csv)
    df, t_col = load_csv(csv_path, args.start, args.end)
    t = df[t_col].to_numpy(dtype=np.float64)
    if not args.no_rebase:
        t = t - float(t[0])

    prefix = (
        Path(args.output_prefix)
        if args.output_prefix is not None
        else csv_path.with_name(f"{csv_path.stem}_smooth_w{args.window_size}")
    )

    fig_ctrl = plot_group(
        t,
        df,
        CTRL_COLS,
        window_size=args.window_size,
        title="Control Commands",
        ylabel="ctrl",
        start=args.start,
        end=args.end,
        rebased=not args.no_rebase,
    )
    fig_flow = plot_group(
        t,
        df,
        FLOW_COLS,
        window_size=args.window_size,
        title="Flow Q",
        ylabel="Q",
        start=args.start,
        end=args.end,
        rebased=not args.no_rebase,
    )

    ctrl_path = prefix.with_name(f"{prefix.name}_ctrl.png")
    flow_path = prefix.with_name(f"{prefix.name}_flow.png")
    fig_ctrl.savefig(ctrl_path, dpi=args.dpi)
    fig_flow.savefig(flow_path, dpi=args.dpi)

    print(f"[INFO] rows plotted: {len(df)}")
    print(f"[INFO] ctrl plot saved: {ctrl_path}")
    print(f"[INFO] flow plot saved: {flow_path}")

    if args.no_show:
        plt.close(fig_ctrl)
        plt.close(fig_flow)
    else:
        plt.show()


if __name__ == "__main__":
    main()
