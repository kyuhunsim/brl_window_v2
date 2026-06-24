#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from pneu_tune.sim4 import PneuSim


ATM_KPA = 101.325


def _time_col(df: pd.DataFrame) -> str:
    if "time" in df.columns:
        return "time"
    if "curr_time" in df.columns:
        return "curr_time"
    raise ValueError("CSV needs time or curr_time column")


def _window_mask(time: np.ndarray, start: float | None, end: float | None) -> np.ndarray:
    mask = np.ones(time.shape, dtype=bool)
    if start is not None:
        mask &= time >= float(start)
    if end is not None:
        mask &= time <= float(end)
    return mask


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def _describe_window(label: str, df: pd.DataFrame, mask: np.ndarray) -> None:
    if not np.any(mask):
        print(f"{label:28s}: no samples")
        return
    g = df.loc[mask]
    print(
        f"{label:28s}: t={g['_t'].iloc[0]:.3f}..{g['_t'].iloc[-1]:.3f}s "
        f"n={len(g)} angle={g['angle'].mean():.6f}+/-{g['angle'].std():.6f} deg "
        f"act_pos={g['act_pos_press'].mean():.3f} kPa "
        f"act_neg={g['act_neg_press'].mean():.3f} kPa "
        f"dP={g['_act_dp'].mean():.3f} kPa"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate initial total length from two angles: theta_ext at near-zero "
            "actuator differential pressure, and theta_init at the actual trajectory start."
        )
    )
    ap.add_argument("csv", type=Path, help="experiment CSV path")
    ap.add_argument("--freq", type=float, default=50.0, help="sim geometry frequency")
    ap.add_argument("--init-start", type=float, default=0.0, help="theta_init window start [sec]")
    ap.add_argument("--init-end", type=float, default=20.0, help="theta_init window end [sec]")
    ap.add_argument("--ext-start", type=float, default=None, help="optional theta_ext search/window start [sec]")
    ap.add_argument("--ext-end", type=float, default=None, help="optional theta_ext search/window end [sec]")
    ap.add_argument(
        "--ext-dp-kpa",
        type=float,
        default=2.0,
        help="use samples with |act_pos-act_neg| <= this threshold for theta_ext",
    )
    ap.add_argument(
        "--ext-top-frac",
        type=float,
        default=0.02,
        help="if too few near-zero-dP samples exist, use this fraction with smallest |dP|",
    )
    ap.add_argument(
        "--valid-angle-min",
        type=float,
        default=1.0,
        help="ignore encoder dropout samples at or below this angle [deg]",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"act_pos_press", "act_neg_press", "angle"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{args.csv}: missing required columns: {missing}")

    tcol = _time_col(df)
    df = df.sort_values(tcol).reset_index(drop=True)
    df["_t"] = df[tcol].to_numpy(dtype=np.float64) - float(df[tcol].iloc[0])
    df["_act_dp"] = df["act_pos_press"].to_numpy(dtype=np.float64) - df["act_neg_press"].to_numpy(dtype=np.float64)
    df["_abs_dp"] = np.abs(df["_act_dp"].to_numpy(dtype=np.float64))

    valid = (
        np.isfinite(df["_t"].to_numpy(dtype=np.float64))
        & np.isfinite(df["angle"].to_numpy(dtype=np.float64))
        & (df["angle"].to_numpy(dtype=np.float64) > float(args.valid_angle_min))
        & np.isfinite(df["act_pos_press"].to_numpy(dtype=np.float64))
        & np.isfinite(df["act_neg_press"].to_numpy(dtype=np.float64))
    )
    df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("no valid samples after filtering")

    time = df["_t"].to_numpy(dtype=np.float64)
    init_mask = _window_mask(time, args.init_start, args.init_end)
    if not np.any(init_mask):
        raise SystemExit(f"init window has no valid samples: {args.init_start}..{args.init_end} sec")

    ext_window = _window_mask(time, args.ext_start, args.ext_end)
    ext_candidates = ext_window & (df["_abs_dp"].to_numpy(dtype=np.float64) <= float(args.ext_dp_kpa))
    min_count = max(10, int(0.25 * float(args.freq)))
    if int(np.sum(ext_candidates)) < min_count:
        idx = np.flatnonzero(ext_window)
        if idx.size == 0:
            raise SystemExit("theta_ext search window has no valid samples")
        n = max(min_count, int(np.ceil(float(args.ext_top_frac) * idx.size)))
        order = idx[np.argsort(df["_abs_dp"].to_numpy(dtype=np.float64)[idx])[:n]]
        ext_candidates = np.zeros(len(df), dtype=bool)
        ext_candidates[order] = True

    theta_init, theta_init_std = _mean_std(df.loc[init_mask, "angle"].to_numpy(dtype=np.float64))
    theta_ext, theta_ext_std = _mean_std(df.loc[ext_candidates, "angle"].to_numpy(dtype=np.float64))

    sim = PneuSim(freq=float(args.freq), delay=0.0, noise=False, scale=False)
    contraction_init = float(sim.angle_to_displacement(theta_init, theta_ext))
    l_init_total = float(sim.actuator_total_initial_length - contraction_init)

    # This range is diagnostic only; l_init_total may be outside if theta_ext is poorly identified.
    length_delta = np.asarray(sim.angle_to_length_delta(df["angle"].to_numpy(dtype=np.float64), theta_init), dtype=np.float64)
    l_lower = float(sim.actuator_min_total_length - np.min(length_delta))
    l_upper = float(sim.actuator_total_initial_length - np.max(length_delta))

    print(f"csv                         : {args.csv}")
    print(f"time range                  : 0.000 .. {float(time[-1]):.3f} s")
    print(f"valid samples               : {len(df)}")
    print()
    _describe_window("theta_init window", df, init_mask)
    _describe_window("theta_ext samples", df, ext_candidates)
    print()
    print(f"theta_init                  : {theta_init:.6f} +/- {theta_init_std:.6f} deg")
    print(f"theta_ext                   : {theta_ext:.6f} +/- {theta_ext_std:.6f} deg")
    print(f"theta_ext - theta_init       : {theta_ext - theta_init:.6f} deg")
    print(f"init contraction from ext    : {1000.0 * contraction_init:.3f} mm")
    print(f"estimated l_init_total       : {l_init_total:.12g} m ({1000.0 * l_init_total:.3f} mm)")
    print(f"diagnostic valid range       : {l_lower:.12g} .. {l_upper:.12g} m ({1000.0*l_lower:.3f} .. {1000.0*l_upper:.3f} mm)")
    if not (l_lower <= l_init_total <= l_upper):
        print("warning                     : l_init_total is outside diagnostic range; check theta_ext selection")
    print()
    print("use with tune4:")
    print(f"  --init l_init_total={l_init_total:.16g}")
    print("  and exclude l_init_total from --params")


if __name__ == "__main__":
    main()
