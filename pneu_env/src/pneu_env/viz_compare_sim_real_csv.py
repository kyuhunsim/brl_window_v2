#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pneu_utils.utils import setup_plot_style

# ============================================================
# FIXED SETTINGS
# ============================================================
START_TIME = None
END_TIME = None
REBASE_TIME = True

SHOW_FIGURE = True
SAVE_FIGURE = False
SAVE_PATH_PRESSURE = "compare_pressure.png"
SAVE_PATH_FLOW = "compare_flow6.png"

FIG_WIDTH = 22.0
FIG_HEIGHT_PRESSURE = 12.0
FIG_HEIGHT_FLOW = 18.0


# ============================================================
# HELPERS
# ============================================================
def _require_columns(df: pd.DataFrame, required_cols: list[str], path: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Got: {list(df.columns)}")


def _resolve_sim_flow_columns(df_sim: pd.DataFrame, sim_csv: str) -> list[str]:
    calc_cols = [f"calc_flow{i}" for i in range(1, 7)]
    mf_cols = [f"mf_flow{i}" for i in range(1, 7)]
    if all(c in df_sim.columns for c in calc_cols):
        return calc_cols
    if all(c in df_sim.columns for c in mf_cols):
        return mf_cols
    raise ValueError(
        f"Missing sim flow columns in {sim_csv}. "
        f"Need {calc_cols} or {mf_cols}. Got: {list(df_sim.columns)}"
    )


def _resolve_sim_pressure_columns(df_sim: pd.DataFrame, sim_csv: str) -> tuple[dict[str, str], str]:
    sim_cols = {
        "press_pos": "sim_press_pos",
        "press_neg": "sim_press_neg",
        "act_pos_press": "sim_act_pos_press",
        "act_neg_press": "sim_act_neg_press",
    }
    legacy_cols = {
        "press_pos": "press_pos",
        "press_neg": "press_neg",
        "act_pos_press": "act_pos_press",
        "act_neg_press": "act_neg_press",
    }
    if all(col in df_sim.columns for col in sim_cols.values()):
        return sim_cols, "sim_press_*"
    if all(col in df_sim.columns for col in legacy_cols.values()):
        return legacy_cols, "press_*"
    raise ValueError(
        f"Missing sim pressure columns in {sim_csv}. "
        f"Need {list(sim_cols.values())} or {list(legacy_cols.values())}. "
        f"Got: {list(df_sim.columns)}"
    )


def _load_csv(
    path: str,
    *,
    required_cols: list[str],
    start: Optional[float],
    end: Optional[float],
    rebase_time: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "curr_time" not in df.columns:
        if "time" not in df.columns:
            raise ValueError(f"Missing time column(curr_time/time) in {path}. Got: {list(df.columns)}")
        df = df.rename(columns={"time": "curr_time"})

    _require_columns(df, required_cols, path)

    df = df.sort_values("curr_time").reset_index(drop=True)

    if start is not None:
        df = df[df["curr_time"] >= float(start)]
    if end is not None:
        df = df[df["curr_time"] <= float(end)]

    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No data points after time filtering: {path}")

    if rebase_time:
        df["curr_time"] = df["curr_time"] - float(df["curr_time"].iloc[0])

    return df


def _clean_xy(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if t.size == 0:
        return t, y

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    uniq_t, inv = np.unique(t, return_inverse=True)
    if uniq_t.size != t.size:
        sums = np.zeros_like(uniq_t, dtype=np.float64)
        counts = np.zeros_like(uniq_t, dtype=np.float64)
        np.add.at(sums, inv, y)
        np.add.at(counts, inv, 1.0)
        y = sums / np.maximum(counts, 1.0)
        t = uniq_t

    return t, y


def _align_on_overlap(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_other: np.ndarray,
    y_other: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    t_ref, y_ref = _clean_xy(t_ref, y_ref)
    t_other, y_other = _clean_xy(t_other, y_other)

    if t_ref.size < 2 or t_other.size < 2:
        return None

    t0 = max(float(t_ref[0]), float(t_other[0]))
    t1 = min(float(t_ref[-1]), float(t_other[-1]))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None

    mask = (t_ref >= t0) & (t_ref <= t1)
    t_common = t_ref[mask]
    if t_common.size < 2:
        return None

    y_ref_common = y_ref[mask]
    y_other_interp = np.interp(t_common, t_other, y_other)
    return y_ref_common, y_other_interp


def _metrics_time_aligned(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_other: np.ndarray,
    y_other: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    aligned = _align_on_overlap(t_ref, y_ref, t_other, y_other)
    if aligned is None:
        return dict(rmse=float("nan"), mae=float("nan"), acc=float("nan"))

    y_ref_common, y_other_interp = aligned
    err = y_ref_common - y_other_interp

    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    denom = np.maximum(np.abs(y_ref_common), eps)
    mape = float(np.mean(np.abs(err) / denom)) if err.size else float("nan")
    acc = 1.0 - mape if np.isfinite(mape) else float("nan")
    return dict(rmse=rmse, mae=mae, acc=acc)


def _fmt_num(v: float, fmt: str = ".4g") -> str:
    if not np.isfinite(v):
        return "nan"
    return format(v, fmt)


def _fmt_pct(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v * 100:.1f}%"


def _robust_ylim(*series: np.ndarray):
    vals = np.concatenate([np.asarray(s, dtype=np.float64) for s in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        pad = 1.0 if lo == 0 else 0.05 * abs(lo)
        return lo - pad, hi + pad

    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


# ============================================================
# STYLE
# ============================================================
setup_plot_style({"legend.fontsize": 18})

LW_REAL = 3.2
LW_SIM = 2.4
LW_GRID_MAJOR = 0.9
GRID_ALPHA_MAJOR = 0.35

C_SIM = "#ff0000"
C_REAL = "#0047ff"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="sim csv path")
    ap.add_argument("--real", required=True, help="real csv path")
    ap.add_argument("--sim-start", type=float, default=START_TIME, help="sim csv start time filter")
    ap.add_argument("--sim-end", type=float, default=END_TIME, help="sim csv end time filter")
    ap.add_argument("--real-start", type=float, default=START_TIME, help="real csv start time filter")
    ap.add_argument("--real-end", type=float, default=END_TIME, help="real csv end time filter")
    args = ap.parse_args()

    sim_csv = args.sim
    real_csv = args.real

    sim_required = ["curr_time"]
    real_required = [
        "curr_time",
        "press_pos",
        "press_neg",
        "act_pos_press",
        "act_neg_press",
        "flow1",
        "flow2",
        "flow3",
        "flow4",
        "flow5",
        "flow6",
    ]

    df_sim = _load_csv(
        sim_csv,
        required_cols=sim_required,
        start=args.sim_start,
        end=args.sim_end,
        rebase_time=REBASE_TIME,
    )
    df_real = _load_csv(
        real_csv,
        required_cols=real_required,
        start=args.real_start,
        end=args.real_end,
        rebase_time=REBASE_TIME,
    )

    t_sim = df_sim["curr_time"].to_numpy(dtype=np.float64)
    t_real = df_real["curr_time"].to_numpy(dtype=np.float64)
    sim_flow_cols = _resolve_sim_flow_columns(df_sim, sim_csv)
    sim_press_cols, sim_press_source = _resolve_sim_pressure_columns(df_sim, sim_csv)
    if sim_press_source == "press_*":
        print(
            "[WARN] Using press_* columns from sim csv for pressure comparison. "
            "If this csv was generated by older valve replay scripts, these may be copied "
            "from real input pressures."
        )
    else:
        print("[INFO] Using sim_press_* columns for pressure comparison.")

    # -------------------------
    # Pressure
    # -------------------------
    sim_press_pos = df_sim[sim_press_cols["press_pos"]].to_numpy(dtype=np.float64)
    sim_press_neg = df_sim[sim_press_cols["press_neg"]].to_numpy(dtype=np.float64)
    sim_act_pos = df_sim[sim_press_cols["act_pos_press"]].to_numpy(dtype=np.float64)
    sim_act_neg = df_sim[sim_press_cols["act_neg_press"]].to_numpy(dtype=np.float64)

    real_press_pos = df_real["press_pos"].to_numpy(dtype=np.float64)
    real_press_neg = df_real["press_neg"].to_numpy(dtype=np.float64)
    real_act_pos = df_real["act_pos_press"].to_numpy(dtype=np.float64)
    real_act_neg = df_real["act_neg_press"].to_numpy(dtype=np.float64)

    press_metrics = {
        "press_pos": _metrics_time_aligned(t_real, real_press_pos, t_sim, sim_press_pos),
        "press_neg": _metrics_time_aligned(t_real, real_press_neg, t_sim, sim_press_neg),
        "act_pos": _metrics_time_aligned(t_real, real_act_pos, t_sim, sim_act_pos),
        "act_neg": _metrics_time_aligned(t_real, real_act_neg, t_sim, sim_act_neg),
    }

    # -------------------------
    # Flow1 ~ Flow6
    # -------------------------
    flow_series = {}
    flow_metrics = {}

    for i in range(1, 7):
        real_name = f"flow{i}"
        sim_name = sim_flow_cols[i - 1]
        real_vals = df_real[real_name].to_numpy(dtype=np.float64)
        sim_vals = df_sim[sim_name].to_numpy(dtype=np.float64)

        flow_series[i] = {
            "real_name": real_name,
            "real_vals": real_vals,
            "sim_name": sim_name,
            "sim_vals": sim_vals,
        }
        flow_metrics[i] = _metrics_time_aligned(t_real, real_vals, t_sim, sim_vals)

    print("[INFO] Pressure metrics (real vs sim, time-aligned on overlap):")
    for key in ["press_pos", "press_neg", "act_pos", "act_neg"]:
        m = press_metrics[key]
        print(
            f"  - {key}: MAE={_fmt_num(m['mae'])} "
            f"RMSE={_fmt_num(m['rmse'])} "
            f"1-MAPE={_fmt_pct(m['acc'])}"
        )

    print("[INFO] Flow1~6 metrics (real vs sim, time-aligned on overlap):")
    for i in range(1, 7):
        m = flow_metrics[i]
        print(
            f"  - flow{i} <- {flow_series[i]['sim_name']}: "
            f"MAE={_fmt_num(m['mae'])} "
            f"RMSE={_fmt_num(m['rmse'])} "
            f"1-MAPE={_fmt_pct(m['acc'])}"
        )

    label_sim = os.path.basename(sim_csv)
    label_real = os.path.basename(real_csv)
    title = f"real: {label_real}  |  sim: {label_sim}"

    # ============================================================
    # PLOT 1: PRESSURE
    # ============================================================
    fig_press, axes = plt.subplots(
        4, 1, figsize=(FIG_WIDTH, FIG_HEIGHT_PRESSURE),
        sharex=True, constrained_layout=True
    )
    fig_press.suptitle(f"{title} | Plot 1 (Pressure)")

    axes[0].plot(t_real, real_press_pos, label="real", linewidth=LW_REAL, color=C_REAL, alpha=0.95)
    axes[0].plot(t_sim, sim_press_pos, label="sim", linewidth=LW_SIM, color=C_SIM, alpha=0.9)
    axes[0].set_ylabel("Ch+ [kPa]")
    axes[0].grid(True, alpha=GRID_ALPHA_MAJOR, linewidth=LW_GRID_MAJOR)
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title(f"Ch+ MAE={_fmt_num(press_metrics['press_pos']['mae'])}")

    axes[1].plot(t_real, real_press_neg, label="real", linewidth=LW_REAL, color=C_REAL, alpha=0.95)
    axes[1].plot(t_sim, sim_press_neg, label="sim", linewidth=LW_SIM, color=C_SIM, alpha=0.9)
    axes[1].set_ylabel("Ch- [kPa]")
    axes[1].grid(True, alpha=GRID_ALPHA_MAJOR, linewidth=LW_GRID_MAJOR)
    axes[1].legend(loc="upper right", ncol=2)
    axes[1].set_title(f"Ch- MAE={_fmt_num(press_metrics['press_neg']['mae'])}")

    axes[2].plot(t_real, real_act_pos, label="real", linewidth=LW_REAL, color=C_REAL, alpha=0.95)
    axes[2].plot(t_sim, sim_act_pos, label="sim", linewidth=LW_SIM, color=C_SIM, alpha=0.9)
    axes[2].set_ylabel("Act+ [kPa]")
    axes[2].grid(True, alpha=GRID_ALPHA_MAJOR, linewidth=LW_GRID_MAJOR)
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].set_title(f"Act+ MAE={_fmt_num(press_metrics['act_pos']['mae'])}")

    axes[3].plot(t_real, real_act_neg, label="real", linewidth=LW_REAL, color=C_REAL, alpha=0.95)
    axes[3].plot(t_sim, sim_act_neg, label="sim", linewidth=LW_SIM, color=C_SIM, alpha=0.9)
    axes[3].set_ylabel("Act- [kPa]")
    axes[3].grid(True, alpha=GRID_ALPHA_MAJOR, linewidth=LW_GRID_MAJOR)
    axes[3].legend(loc="upper right", ncol=2)
    axes[3].set_title(f"Act- MAE={_fmt_num(press_metrics['act_neg']['mae'])}")
    axes[3].set_xlabel("Time [sec]")

    for ax, ys in zip(
        axes,
        [
            (real_press_pos, sim_press_pos),
            (real_press_neg, sim_press_neg),
            (real_act_pos, sim_act_pos),
            (real_act_neg, sim_act_neg),
        ],
    ):
        ylim = _robust_ylim(*ys)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # ============================================================
    # PLOT 2: FLOW 1~6
    # ============================================================
    fig_flow, axes_flow = plt.subplots(
        6, 1, figsize=(FIG_WIDTH, FIG_HEIGHT_FLOW),
        sharex=True, constrained_layout=True
    )
    fig_flow.suptitle(f"{title} | Plot 2 (Flow1~Flow6)")

    flow_labels = {
        1: "Flow1 (Chamber+ valve)",
        2: "Flow2 (Chamber- valve)",
        3: "Flow3 (Act+ in)",
        4: "Flow4 (Act+ out)",
        5: "Flow5 (Act- in)",
        6: "Flow6 (Act- out)",
    }

    for i, ax in enumerate(axes_flow, start=1):
        real_vals = flow_series[i]["real_vals"]
        sim_vals = flow_series[i]["sim_vals"]
        sim_name = flow_series[i]["sim_name"]
        m = flow_metrics[i]

        ax.plot(t_real, real_vals, label=f"real flow{i}", linewidth=LW_REAL, color=C_REAL, alpha=0.95)
        ax.plot(t_sim, sim_vals, label=f"sim ({sim_name})", linewidth=LW_SIM, color=C_SIM, alpha=0.9)

        ax.set_ylabel("L/min")
        ax.grid(True, alpha=GRID_ALPHA_MAJOR, linewidth=LW_GRID_MAJOR)
        ax.legend(loc="upper right", ncol=2)
        ax.set_title(f"{flow_labels[i]}  |  MAE={_fmt_num(m['mae'])}")

        ylim = _robust_ylim(real_vals, sim_vals)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes_flow[-1].set_xlabel("Time [sec]")

    for ax in list(axes) + list(axes_flow):
        ax.yaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontweight("bold")
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(15)
                text.set_fontweight("bold")

    if SAVE_FIGURE:
        fig_press.savefig(SAVE_PATH_PRESSURE, dpi=150)
        fig_flow.savefig(SAVE_PATH_FLOW, dpi=150)
        print(f"[INFO] Saved pressure figure: {SAVE_PATH_PRESSURE}")
        print(f"[INFO] Saved flow figure: {SAVE_PATH_FLOW}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig_press)
        plt.close(fig_flow)


if __name__ == "__main__":
    main()
