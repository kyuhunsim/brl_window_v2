#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze real_chirp_pid_2ctrl.py CSV output.

The script estimates a frequency response from chirp input to measured flow:

    input  = chirp_ctrl by default
    output = flow1/flow2 by default

For each local window it fits:

    y(t) = a*sin(w*t) + b*cos(w*t) + c

to input and output, then computes gain, gain_dB, phase lag, -3 dB bandwidth,
and a second-order low-pass fit:

    G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from pneu_utils.utils import get_pkg_path


WARMUP_EXCLUDE_SEC = 10.0


def fit_sine(time_s: np.ndarray, signal: np.ndarray, freq_hz: float) -> tuple[float, float, float]:
    omega = 2.0 * np.pi * freq_hz
    t = time_s - time_s[0]
    design = np.column_stack([
        np.sin(omega * t),
        np.cos(omega * t),
        np.ones_like(t),
    ])
    coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
    a, b, c = coef
    amp = float(np.hypot(a, b))
    phase = float(np.arctan2(b, a))
    return amp, phase, float(c)


def second_order_response(freq_hz: np.ndarray, wn: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * freq_hz
    mag = wn * wn / np.sqrt((wn * wn - omega * omega) ** 2 + (2.0 * zeta * wn * omega) ** 2)
    phase = -np.arctan2(2.0 * zeta * wn * omega, wn * wn - omega * omega)
    return mag, phase


def estimate_windows(
    df: pd.DataFrame,
    *,
    time_col: str,
    input_col: str,
    output_col: str,
    freq_col: str,
    cycles_per_window: float,
    step_fraction: float,
    min_points: int,
    min_input_amp: float,
) -> pd.DataFrame:
    time_s = df[time_col].to_numpy(dtype=float)
    input_u = df[input_col].to_numpy(dtype=float)
    output_y = df[output_col].to_numpy(dtype=float)
    freq = df[freq_col].to_numpy(dtype=float)

    valid = np.isfinite(time_s) & np.isfinite(input_u) & np.isfinite(output_y) & np.isfinite(freq) & (freq > 0.0)
    time_s = time_s[valid]
    input_u = input_u[valid]
    output_y = output_y[valid]
    freq = freq[valid]

    rows = []
    idx = 0
    n = len(time_s)
    while idx < n:
        f0 = float(freq[idx])
        window_sec = max(cycles_per_window / max(f0, 1e-9), 0.1)
        t_start = time_s[idx]
        t_end = t_start + window_sec
        end_idx = int(np.searchsorted(time_s, t_end, side="right"))
        if end_idx - idx < min_points:
            idx += max(1, min_points // 2)
            continue

        sl = slice(idx, end_idx)
        local_freq = float(np.median(freq[sl]))
        if local_freq <= 0.0:
            idx += 1
            continue

        t_win = time_s[sl]
        u_win = input_u[sl]
        y_win = output_y[sl]

        u_amp, u_phase, u_offset = fit_sine(t_win, u_win, local_freq)
        y_amp, y_phase, y_offset = fit_sine(t_win, y_win, local_freq)
        if u_amp < min_input_amp:
            idx += max(1, int((end_idx - idx) * step_fraction))
            continue

        gain = y_amp / u_amp
        phase = np.arctan2(np.sin(y_phase - u_phase), np.cos(y_phase - u_phase))
        delay_sec = -phase / (2.0 * np.pi * local_freq)

        rows.append(
            dict(
                time_mid=float(0.5 * (t_win[0] + t_win[-1])),
                freq_hz=local_freq,
                input_amp=u_amp,
                output_amp=y_amp,
                input_offset=u_offset,
                output_offset=y_offset,
                gain=gain,
                phase_rad=float(phase),
                phase_deg=float(np.rad2deg(phase)),
                delay_sec=float(delay_sec),
                n_points=int(end_idx - idx),
            )
        )

        step = max(1, int((end_idx - idx) * step_fraction))
        idx += step

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("freq_hz").reset_index(drop=True)
    low_gain = float(out.head(max(1, min(5, len(out))))["gain"].median())
    out["gain_norm"] = out["gain"] / max(low_gain, 1e-12)
    out["gain_db"] = 20.0 * np.log10(np.maximum(out["gain_norm"], 1e-12))
    out["phase_unwrapped_rad"] = np.unwrap(out["phase_rad"].to_numpy())
    out["phase_unwrapped_deg"] = np.rad2deg(out["phase_unwrapped_rad"])
    return out


def find_bandwidth(response: pd.DataFrame, threshold_db: float) -> Optional[float]:
    below = response[response["gain_db"] <= threshold_db]
    if below.empty:
        return None
    return float(below.iloc[0]["freq_hz"])


def fit_second_order(response: pd.DataFrame, *, use_phase: bool) -> dict:
    freq = response["freq_hz"].to_numpy(dtype=float)
    gain = response["gain_norm"].to_numpy(dtype=float)
    phase = response["phase_unwrapped_rad"].to_numpy(dtype=float)

    valid = np.isfinite(freq) & np.isfinite(gain) & (freq > 0.0) & (gain > 0.0)
    if use_phase:
        valid &= np.isfinite(phase)
    freq = freq[valid]
    gain = gain[valid]
    phase = phase[valid]
    if len(freq) < 4:
        return dict(success=False, reason="not enough points")

    def residual(log_params):
        wn = np.exp(log_params[0])
        zeta = np.exp(log_params[1])
        mag_model, phase_model = second_order_response(freq, wn, zeta)
        res_gain = 20.0 * np.log10(np.maximum(mag_model, 1e-12)) - 20.0 * np.log10(np.maximum(gain, 1e-12))
        if not use_phase:
            return res_gain
        res_phase = np.rad2deg(phase_model - phase) / 45.0
        return np.concatenate([res_gain, res_phase])

    bw_guess = find_bandwidth(response, -3.0)
    if bw_guess is None:
        bw_guess = float(np.median(freq))
    wn0 = 2.0 * np.pi * max(bw_guess, 1e-3)
    zeta0 = 0.8

    res = optimize.least_squares(
        residual,
        x0=np.log([wn0, zeta0]),
        bounds=(
            np.log([2.0 * np.pi * 1e-3, 0.05]),
            np.log([2.0 * np.pi * 100.0, 20.0]),
        ),
        max_nfev=5000,
    )
    wn = float(np.exp(res.x[0]))
    zeta = float(np.exp(res.x[1]))
    return dict(
        success=bool(res.success),
        cost=float(res.cost),
        wn_rad_s=wn,
        wn_hz=wn / (2.0 * np.pi),
        zeta=zeta,
        message=str(res.message),
    )


def plot_results(
    df: pd.DataFrame,
    response: pd.DataFrame,
    fit: dict,
    *,
    time_col: str,
    input_col: str,
    output_col: str,
    freq_col: str,
    bandwidth_hz: Optional[float],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    axes[0].plot(df[time_col], df[input_col], label=input_col, color="black")
    axes[0].set_ylabel("Input")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    ax0b = axes[0].twinx()
    ax0b.plot(df[time_col], df[freq_col], label=freq_col, color="tab:gray", alpha=0.45)
    ax0b.set_ylabel("Freq [Hz]")

    axes[1].plot(df[time_col], df[output_col], label=output_col, color="tab:blue")
    axes[1].set_ylabel("Output flow")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].scatter(response["freq_hz"], response["gain_db"], s=18, label="measured", color="tab:red")
    axes[2].axhline(-3.0, color="gray", linestyle="--", linewidth=1.0, label="-3 dB")
    if bandwidth_hz is not None:
        axes[2].axvline(bandwidth_hz, color="tab:purple", linestyle="--", linewidth=1.0, label=f"BW={bandwidth_hz:.3g} Hz")

    if fit.get("success"):
        f_model = np.logspace(np.log10(max(response["freq_hz"].min(), 1e-4)), np.log10(response["freq_hz"].max()), 300)
        mag, phase_model = second_order_response(f_model, fit["wn_rad_s"], fit["zeta"])
        axes[2].plot(f_model, 20.0 * np.log10(np.maximum(mag, 1e-12)), color="black", label="2nd-order fit")
        axes[3].plot(f_model, np.rad2deg(phase_model), color="black", label="2nd-order fit")

    axes[2].set_xscale("log")
    axes[2].set_ylabel("Gain [dB]")
    axes[2].grid(True, which="both")
    axes[2].legend(loc="best")

    axes[3].scatter(response["freq_hz"], response["phase_unwrapped_deg"], s=18, label="measured", color="tab:green")
    axes[3].set_xscale("log")
    axes[3].set_ylabel("Phase [deg]")
    axes[3].set_xlabel("Frequency [Hz]")
    axes[3].grid(True, which="both")
    axes[3].legend(loc="best")

    title = "Chirp flow analysis"
    if fit.get("success"):
        title += f" | wn={fit['wn_hz']:.3g} Hz ({fit['wn_rad_s']:.3g} rad/s), zeta={fit['zeta']:.3g}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV from real_chirp_pid_2ctrl.py")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--input-col", default="chirp_ctrl")
    parser.add_argument("--output-col", default="flow1")
    parser.add_argument("--freq-col", default="chirp_freq")
    parser.add_argument("--profile-mode-col", default=None)
    parser.add_argument("--profile-mode", type=float, default=None)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--cycles-per-window", type=float, default=4.0)
    parser.add_argument("--step-fraction", type=float, default=0.5)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--min-input-amp", type=float, default=1e-4)
    parser.add_argument("--threshold-db", type=float, default=-3.0)
    parser.add_argument("--no-phase-fit", action="store_true")
    parser.add_argument("--save-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    if args.profile_mode_col and args.profile_mode is not None:
        df = df[df[args.profile_mode_col] == args.profile_mode].copy()
    if df.empty:
        raise RuntimeError("No rows after profile mode filtering.")

    warmup_cutoff = float(df[args.time_col].min()) + WARMUP_EXCLUDE_SEC
    df = df[df[args.time_col] >= warmup_cutoff].copy()
    if args.start is not None:
        df = df[df[args.time_col] >= args.start].copy()
    if args.end is not None:
        df = df[df[args.time_col] <= args.end].copy()
    if df.empty:
        raise RuntimeError("No rows after time filtering.")

    response = estimate_windows(
        df,
        time_col=args.time_col,
        input_col=args.input_col,
        output_col=args.output_col,
        freq_col=args.freq_col,
        cycles_per_window=args.cycles_per_window,
        step_fraction=args.step_fraction,
        min_points=args.min_points,
        min_input_amp=args.min_input_amp,
    )
    if response.empty:
        raise RuntimeError("No valid chirp response windows. Check columns, frequency, and signal amplitude.")

    bandwidth_hz = find_bandwidth(response, args.threshold_db)
    fit = fit_second_order(response, use_phase=not args.no_phase_fit)

    if args.save_name:
        save_name = args.save_name
    else:
        stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        stem = os.path.splitext(os.path.basename(args.csv))[0]
        save_name = f"{stamp}_{stem}_chirp_analysis"

    out_dir = os.path.join(get_pkg_path("pneu_env"), "exp", save_name)
    os.makedirs(out_dir, exist_ok=True)

    response_path = os.path.join(out_dir, "chirp_response.csv")
    response.to_csv(response_path, index=False)

    summary = dict(
        csv=os.path.abspath(args.csv),
        input_col=args.input_col,
        output_col=args.output_col,
        freq_col=args.freq_col,
        n_windows=int(len(response)),
        bandwidth_hz=bandwidth_hz,
        threshold_db=args.threshold_db,
        warmup_exclude_sec=WARMUP_EXCLUDE_SEC,
        warmup_cutoff=warmup_cutoff,
        fit=fit,
        args=vars(args),
    )
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_path = os.path.join(out_dir, "chirp_analysis.png")
    plot_results(
        df,
        response,
        fit,
        time_col=args.time_col,
        input_col=args.input_col,
        output_col=args.output_col,
        freq_col=args.freq_col,
        bandwidth_hz=bandwidth_hz,
        output_path=plot_path,
    )

    print(f"[INFO] Saved: {out_dir}")
    print(f"[INFO] Excluded initial {WARMUP_EXCLUDE_SEC:g} sec before analysis (time < {warmup_cutoff:.6g}).")
    print(f"[INFO] Response CSV: {response_path}")
    print(f"[INFO] Summary JSON: {summary_path}")
    if bandwidth_hz is None:
        print("[INFO] Bandwidth: not reached")
    else:
        print(f"[INFO] Bandwidth ({args.threshold_db:g} dB): {bandwidth_hz:.6g} Hz")
    if fit.get("success"):
        print(
            "[INFO] Fit: "
            f"wn={fit['wn_rad_s']:.6g} rad/s ({fit['wn_hz']:.6g} Hz), "
            f"zeta={fit['zeta']:.6g}"
        )
    else:
        print(f"[WARN] Fit failed: {fit}")


if __name__ == "__main__":
    main()
