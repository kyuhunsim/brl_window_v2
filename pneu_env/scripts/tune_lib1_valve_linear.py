#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tune the legacy lib1 static solenoid valve model from real flow CSV data.

The fitted model matches pneu_env/src/pneu_env/lib/pneumatic_CT.cpp:

    current = 0.165 * signal
    Cdkx = max(c1*current - c2 + c3*pressure_term, 0)
    mdot = Cdkx * D * pi * Phi(Pin, Pout)

The script fits ctrl1/flow1 and ctrl2/flow2 by default, saves plots, and
prints C++ snippets that can be pasted back into lib/pneumatic_CT.cpp.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pneu_utils.utils import get_pkg_path


ATM = 101.325
K = 1.4
R = 0.287
T_OUT = 293.15
STD_RHO = 1.20411831637462
SV_D = 1.6
PI = np.pi


VALVE_CONFIGS = {
    1: dict(
        name="ctrl1 / chamber pos -> ATM",
        type="pos",
        ctrl_cols=("ctrl1", "ctrl_pos", "pos_ctrl"),
        flow_cols=("flow1", "flowrate1"),
        p_in_cols=("press_pos", "sen_pos", "pos_press"),
        p_out_cols=("ATM",),
        default=[0.001114363881476042, 0.00015932275143274647, 0.0],
        cpp_names=("Cpos1", "Cpos2", "Cpos3"),
    ),
    2: dict(
        name="ctrl2 / ATM -> chamber neg",
        type="neg",
        ctrl_cols=("ctrl2", "ctrl_neg", "neg_ctrl"),
        flow_cols=("flow2", "flowrate2"),
        p_in_cols=("ATM",),
        p_out_cols=("press_neg", "sen_neg", "neg_press"),
        default=[0.001133425704176126, 0.0001655487332160521, 3.6481684199321504e-08],
        cpp_names=("Cneg1", "Cneg2", "Cneg3"),
    ),
}


def pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_array(df: pd.DataFrame, candidates: Tuple[str, ...], default: Optional[float] = None) -> np.ndarray:
    col = pick_col(df, candidates)
    if col is not None:
        return df[col].to_numpy(dtype=np.float64)
    if default is None:
        raise KeyError(f"None of these columns exist: {candidates}")
    return np.full(len(df), float(default), dtype=np.float64)


def maybe_smooth(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    window = int(window)
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def maybe_shift_flow(flow: np.ndarray, delay_samples: int) -> np.ndarray:
    """Positive delay_samples means measured flow lags the command, so compare model[:-d] to real[d:]."""
    flow = np.asarray(flow, dtype=np.float64)
    delay_samples = int(delay_samples)
    if delay_samples == 0:
        return flow
    shifted = np.full_like(flow, np.nan)
    if delay_samples > 0:
        shifted[:-delay_samples] = flow[delay_samples:]
    else:
        d = abs(delay_samples)
        shifted[d:] = flow[:-d]
    return shifted


def compressible_phi(P_in_kpa: np.ndarray, P_out_kpa: np.ndarray) -> np.ndarray:
    pin = 1000.0 * np.asarray(P_in_kpa, dtype=np.float64)
    pout = 1000.0 * np.asarray(P_out_kpa, dtype=np.float64)
    phi = np.zeros_like(pin, dtype=np.float64)
    valid = pin >= pout
    if not np.any(valid):
        return phi

    pcr = (2.0 / (K + 1.0)) ** (K / (K - 1.0))
    ratio = np.zeros_like(pin, dtype=np.float64)
    ratio[valid] = np.clip(pout[valid] / np.maximum(pin[valid], 1e-12), 0.0, 1.0)

    choked = valid & (ratio <= pcr)
    sub = valid & (ratio > pcr)
    phi[choked] = (
        pin[choked] / np.sqrt(1000.0 * R * T_OUT)
        * np.sqrt(K * (2.0 / (K + 1.0)) ** ((K + 1.0) / (K - 1.0)))
    )
    phi[sub] = (
        pin[sub] / np.sqrt(1000.0 * R * T_OUT)
        * np.sqrt(2.0 * K / (K - 1.0))
        * np.sqrt(np.maximum(ratio[sub] ** (2.0 / K) - ratio[sub] ** ((K + 1.0) / K), 0.0))
    )
    return phi


def valve_flow_lpm(params: np.ndarray, ctrl: np.ndarray, p_in: np.ndarray, p_out: np.ndarray, valve_type: str) -> np.ndarray:
    c1, c2, c3 = np.asarray(params, dtype=np.float64)
    current = 0.165 * np.asarray(ctrl, dtype=np.float64)

    pin_pa = 1000.0 * np.asarray(p_in, dtype=np.float64)
    patm_pa = 1000.0 * ATM
    diameter_m = SV_D * 0.01
    seat_area = 0.25 * PI * diameter_m * diameter_m

    if valve_type == "pos":
        pressure_term = (pin_pa - patm_pa) * seat_area
    elif valve_type == "neg":
        # Keep the same legacy lib1 expression. For ctrl2, p_in is ATM, so this term is usually zero.
        pressure_term = (patm_pa - pin_pa) * seat_area
    else:
        raise ValueError(f"unknown valve_type: {valve_type}")

    cdkx = np.maximum(c1 * current - c2 + c3 * pressure_term, 0.0)
    mdot = cdkx * diameter_m * PI * compressible_phi(p_in, p_out)
    return mdot * 60000.0 / STD_RHO


def fit_valve(data: Dict[str, np.ndarray], valve_type: str, initial: List[float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    ctrl = data["ctrl"]
    flow = data["flow"]
    p_in = data["p_in"]
    p_out = data["p_out"]

    valid = (
        np.isfinite(ctrl)
        & np.isfinite(flow)
        & np.isfinite(p_in)
        & np.isfinite(p_out)
        & (ctrl >= 0.0)
        & (ctrl <= 1.0)
        & (flow >= 0.0)
    )
    ctrl = ctrl[valid]
    flow = flow[valid]
    p_in = p_in[valid]
    p_out = p_out[valid]

    def residual(raw_params: np.ndarray) -> np.ndarray:
        pred = valve_flow_lpm(raw_params, ctrl, p_in, p_out, valve_type)
        return pred - flow

    result = least_squares(
        residual,
        x0=np.asarray(initial, dtype=np.float64),
        bounds=([0.0, 0.0, -1e-5], [0.02, 0.02, 1e-5]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=20000,
    )
    params = result.x
    pred = valve_flow_lpm(params, ctrl, p_in, p_out, valve_type)
    err = pred - flow
    metrics = dict(
        rmse=float(np.sqrt(np.mean(err * err))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        n=int(len(flow)),
        success=bool(result.success),
        cost=float(result.cost),
    )
    fit_data = dict(ctrl=ctrl, flow=flow, p_in=p_in, p_out=p_out, pred=pred, err=err)
    return params, fit_data, metrics


def make_plot(out_dir: str, save_name: str, results: Dict[int, Dict[str, object]]) -> str:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4.8 * n), squeeze=False)
    for row, (idx, res) in enumerate(sorted(results.items())):
        cfg = VALVE_CONFIGS[idx]
        data = res["fit_data"]
        metrics = res["metrics"]

        order = np.argsort(data["ctrl"])
        axes[row, 0].plot(data["flow"], color="black", lw=1.2, label="real")
        axes[row, 0].plot(data["pred"], color="tab:red", lw=1.0, label="fit")
        axes[row, 0].set_title(f"Valve {idx}: {cfg['name']} | RMSE={metrics['rmse']:.3f} LPM")
        axes[row, 0].set_ylabel("Flow [LPM]")
        axes[row, 0].legend(loc="upper right")
        axes[row, 0].grid(True, alpha=0.35)

        axes[row, 1].scatter(data["ctrl"], data["flow"], s=4, alpha=0.25, color="black", label="real")
        axes[row, 1].plot(data["ctrl"][order], data["pred"][order], color="tab:red", lw=1.0, label="fit")
        axes[row, 1].set_title("Control vs Flow")
        axes[row, 1].set_xlabel("Control")
        axes[row, 1].set_ylabel("Flow [LPM]")
        axes[row, 1].legend(loc="upper left")
        axes[row, 1].grid(True, alpha=0.35)

        axes[row, 2].hist(data["err"], bins=80, color="tab:blue", alpha=0.8)
        axes[row, 2].axvline(0.0, color="black", lw=1)
        axes[row, 2].set_title(f"Error | MAE={metrics['mae']:.3f}, bias={metrics['bias']:.3f}")
        axes[row, 2].set_xlabel("fit - real [LPM]")
        axes[row, 2].grid(True, alpha=0.35)

    axes[-1, 0].set_xlabel("Sample")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{save_name}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def cpp_snippet(results: Dict[int, Dict[str, object]]) -> str:
    lines = []
    for idx in sorted(results):
        cfg = VALVE_CONFIGS[idx]
        p = results[idx]["params"]
        names = cfg["cpp_names"]
        lines.append(f"// Valve {idx}: {cfg['name']}")
        for name, value in zip(names, p):
            lines.append(f"double {name} = {value:.17g};")
        lines.append("")
    return "\n".join(lines)


def parse_valves(text: str) -> List[int]:
    if str(text).lower() in ("all", "*"):
        return [1, 2]
    valves = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx not in VALVE_CONFIGS:
            raise ValueError(f"Only valves 1 and 2 are supported here, got {idx}")
        valves.append(idx)
    return sorted(set(valves))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune legacy lib1 linear valve model from real flow CSV.")
    parser.add_argument("csv", help="Path to real flow CSV, e.g. exp/xxx.csv")
    parser.add_argument("--valves", default="1,2", help="Valves to tune: 1,2 or all")
    parser.add_argument("--start", type=float, default=None, help="Start time [s]")
    parser.add_argument("--end", type=float, default=None, help="End time [s]")
    parser.add_argument("--flow-window", type=int, default=1, help="Moving average window for real flow")
    parser.add_argument("--flow-delay-samples", type=int, default=0, help="Positive means real flow lags command")
    parser.add_argument("--save-name", default=None, help="Output folder name under pneu_env/tune_result")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(get_pkg_path("pneu_env"), csv_path)
    df = pd.read_csv(csv_path)

    time_col = pick_col(df, ("curr_time", "time", "rt_time"))
    if time_col is not None:
        t = df[time_col].to_numpy(dtype=np.float64)
        mask = np.ones(len(df), dtype=bool)
        if args.start is not None:
            mask &= t >= float(args.start)
        if args.end is not None:
            mask &= t <= float(args.end)
        df = df.loc[mask].reset_index(drop=True)

    save_name = args.save_name
    if save_name is None:
        now = datetime.now().strftime("%y%m%d_%H_%M_%S")
        base = os.path.splitext(os.path.basename(csv_path))[0]
        save_name = f"{now}_{base}_lib1_linear_valve_fit"

    out_dir = os.path.join(get_pkg_path("pneu_env"), "tune_result", save_name)
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[int, Dict[str, object]] = {}
    for idx in parse_valves(args.valves):
        cfg = VALVE_CONFIGS[idx]
        ctrl = get_array(df, cfg["ctrl_cols"])
        flow = get_array(df, cfg["flow_cols"])
        p_in = get_array(df, cfg["p_in_cols"], default=ATM)
        p_out = get_array(df, cfg["p_out_cols"], default=ATM)

        flow = maybe_smooth(flow, args.flow_window)
        flow = maybe_shift_flow(flow, args.flow_delay_samples)

        data = dict(ctrl=ctrl, flow=flow, p_in=p_in, p_out=p_out)
        params, fit_data, metrics = fit_valve(data, cfg["type"], cfg["default"])
        results[idx] = dict(params=params, fit_data=fit_data, metrics=metrics)

        fit_df = pd.DataFrame(
            dict(
                ctrl=fit_data["ctrl"],
                p_in=fit_data["p_in"],
                p_out=fit_data["p_out"],
                real_flow_lpm=fit_data["flow"],
                pred_flow_lpm=fit_data["pred"],
                err_lpm=fit_data["err"],
            )
        )
        fit_df.to_csv(os.path.join(out_dir, f"valve{idx}_fit.csv"), index=False)

    plot_path = make_plot(out_dir, save_name, results)
    snippet = cpp_snippet(results)

    summary = dict(
        csv=csv_path,
        args=vars(args),
        results={
            str(idx): dict(
                params=[float(v) for v in res["params"]],
                metrics=res["metrics"],
                name=VALVE_CONFIGS[idx]["name"],
            )
            for idx, res in results.items()
        },
        cpp_snippet=snippet,
        plot=plot_path,
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "cpp_snippet.txt"), "w", encoding="utf-8") as f:
        f.write(snippet)
        f.write("\n")

    print(f"[INFO] Saved: {out_dir}")
    print(f"[INFO] Plot: {plot_path}")
    print()
    print(snippet)
    print("[INFO] Metrics:")
    for idx, res in sorted(results.items()):
        print(f"  valve{idx}: {res['metrics']}")


if __name__ == "__main__":
    main()
