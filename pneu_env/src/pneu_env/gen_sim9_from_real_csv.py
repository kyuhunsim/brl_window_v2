#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_sim9_from_real_csv.py

Generate a lib9(sim9) replay CSV from a real CSV and include net-flow columns.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sim9 import PneuSim

STD_RHO = 1.20411831637462
DEFAULT_CTRL_SMOOTH_WINDOW = 1
REQUIRED_INPUT_COLUMNS = [
    "press_pos",
    "press_neg",
    "act_pos_press",
    "act_neg_press",
    "ctrl1",
    "ctrl2",
    "ctrl3",
    "ctrl4",
    "ctrl5",
    "ctrl6",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")


def _col(df: pd.DataFrame, primary: str, fallback: Optional[str] = None) -> np.ndarray:
    if primary in df.columns:
        return df[primary].to_numpy(dtype=np.float64)
    if fallback and fallback in df.columns:
        return df[fallback].to_numpy(dtype=np.float64)
    raise ValueError(
        f"Missing column: {primary}"
        + (f" (or fallback {fallback})" if fallback else "")
    )


def _load_real_csv(
    path: str,
    *,
    clip_start_sec: Optional[float],
    clip_end_sec: Optional[float],
    clip_tail_sec: Optional[float],
    rebase_time: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "curr_time" not in df.columns:
        if "time" not in df.columns:
            raise ValueError(f"Missing column: curr_time/time in {path}. Got: {list(df.columns)}")
        df = df.rename(columns={"time": "curr_time"})

    df = df.sort_values("curr_time").reset_index(drop=True)

    t0 = float(df["curr_time"].iloc[0])
    t1 = float(df["curr_time"].iloc[-1])
    start = clip_start_sec
    end = clip_end_sec

    if clip_tail_sec is not None:
        start = max(t0, t1 - float(clip_tail_sec))

    if start is not None:
        df = df[df["curr_time"] >= float(start)]
    if end is not None:
        df = df[df["curr_time"] <= float(end)]

    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows after time clipping.")

    if rebase_time:
        df["curr_time"] = df["curr_time"] - float(df["curr_time"].iloc[0])

    return df


def _row_at_time(
    traj_time: np.ndarray,
    values: np.ndarray,
    t: float,
    idx: int,
) -> Tuple[np.ndarray, int]:
    n = int(traj_time.shape[0])
    while idx + 1 < n and t >= float(traj_time[idx + 1]):
        idx += 1
    return np.asarray(values[idx], dtype=np.float64), idx


def _convert_ctrl_to_unit(values: np.ndarray, domain: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if domain == "unit":
        return np.clip(arr, 0.0, 1.0)
    if domain == "bipolar":
        return 0.5 * (np.clip(arr, -1.0, 1.0) + 1.0)
    raise ValueError(f"Unsupported ctrl domain: {domain}")


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    return (
        pd.Series(arr)
        .rolling(window=int(window), min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _preprocess_ctrl_for_valve_model(
    ctrl_unit_signal: np.ndarray,
    smooth_window: int,
) -> np.ndarray:
    ctrl = np.asarray(ctrl_unit_signal, dtype=np.float64)
    if ctrl.ndim != 2:
        raise ValueError(f"ctrl_unit_signal must be 2D, got shape {ctrl.shape}")

    u_eff = np.clip((ctrl - 0.5) * 2.0, 0.0, 1.0)
    if smooth_window > 1:
        u_eff = np.column_stack(
            [_moving_average_1d(u_eff[:, i], smooth_window) for i in range(u_eff.shape[1])]
        )
    return np.clip(0.5 + 0.5 * u_eff, 0.0, 1.0)


def _to_lpm_from_mdot(mdot: float) -> float:
    return float(mdot * 60000.0 / STD_RHO)


def _mae_rmse(ref: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if ref.size == 0 or pred.size == 0:
        return float("nan"), float("nan")
    e = ref - pred
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def _resolve_default_out_path(real_path: str, out_path: Optional[str]) -> str:
    path = out_path
    if not path:
        base = os.path.splitext(os.path.basename(real_path))[0]
        path = f"{base}_sim9.csv"
    if os.path.splitext(path)[1] == "":
        path = f"{path}.csv"
    return path


def _mode_out_path(base_out_path: str, pre_window: int, sim_window: int) -> str:
    root, ext = os.path.splitext(base_out_path)
    if ext == "":
        ext = ".csv"
    return f"{root}_pre{int(pre_window)}_sim{int(sim_window)}{ext}"


def _build_replay_ctrl(
    ctrl_unit: np.ndarray,
    *,
    pre_smooth_window: int,
) -> np.ndarray:
    if int(pre_smooth_window) > 1:
        replay_ctrl = _preprocess_ctrl_for_valve_model(
            ctrl_unit_signal=ctrl_unit,
            smooth_window=max(0, int(pre_smooth_window)),
        )
        print(
            f"[INFO] Applied valve-effective control smoothing "
            f"(window={int(pre_smooth_window)}) before lib9 replay"
        )
        return replay_ctrl

    print("[INFO] Using exact raw control replay (no pre smoothing)")
    return ctrl_unit.copy()


def _run_single_replay(
    *,
    df: pd.DataFrame,
    traj_time: np.ndarray,
    replay_ctrl: np.ndarray,
    freq: float,
    delay: float,
    scale: bool,
    sim_log: bool,
    sim_ctrl_smooth_window: int,
    no_rebase: bool,
    debug_input_check: bool,
    debug_prefix: str = "",
) -> tuple[pd.DataFrame, float, bool]:
    press_pos = _col(df, "press_pos")
    press_neg = _col(df, "press_neg")
    act_pos_press = _col(df, "act_pos_press")
    act_neg_press = _col(df, "act_neg_press")

    sim = PneuSim(
        freq=float(freq),
        delay=float(delay),
        noise=False,
        scale=bool(scale),
        ctrl_smooth_window=int(sim_ctrl_smooth_window),
        init_pos_press=float(press_pos[0]),
        init_neg_press=float(press_neg[0]),
        init_act_pos_press=float(act_pos_press[0]),
        init_act_neg_press=float(act_neg_press[0]),
    )
    sim.set_logging(bool(sim_log))

    dt = 1.0 / float(freq)
    replay_time = float(traj_time[0])
    t_end = float(traj_time[-1])
    time_offset = float(traj_time[0]) if no_rebase else 0.0

    idx = 0
    dbg_count = 0

    sim_time = []
    sim_press_pos = []
    sim_press_neg = []
    sim_act_pos = []
    sim_act_neg = []
    sim_ctrl = [[] for _ in range(6)]

    calc_flow = [[] for _ in range(6)]
    calc_net_pos = []
    calc_net_neg = []
    mf_act_net_pos = []
    mf_act_net_neg = []

    has_real_flow = all(c in df.columns for c in ["flow3", "flow4", "flow5", "flow6"])
    real_flow3_full = df["flow3"].to_numpy(dtype=np.float64) if has_real_flow else None
    real_flow4_full = df["flow4"].to_numpy(dtype=np.float64) if has_real_flow else None
    real_flow5_full = df["flow5"].to_numpy(dtype=np.float64) if has_real_flow else None
    real_flow6_full = df["flow6"].to_numpy(dtype=np.float64) if has_real_flow else None
    used_real_flow3 = []
    used_real_flow4 = []
    used_real_flow5 = []
    used_real_flow6 = []

    while replay_time < t_end + 1e-12:
        u, idx = _row_at_time(traj_time, replay_ctrl, replay_time, idx)
        if debug_input_check and dbg_count < 15:
            prefix = f"{debug_prefix} " if debug_prefix else ""
            print(
                f"[DBG] {prefix}replay_time={replay_time:.6f} src_idx={idx} "
                f"src_csv_time={traj_time[idx]:.6f} "
                f"u={np.array2string(u, precision=4, suppress_small=False)}"
            )
            dbg_count += 1

        obs, _ = sim.observe(u)
        obs_time = float(obs[0]) + time_offset
        mf = sim.get_mass_flowrate_dict()
        net = sim.get_act_net_flowrate()

        f1 = _to_lpm_from_mdot(mf["chamber_pos_valve"])
        f2 = _to_lpm_from_mdot(mf["chamber_neg_valve"])
        f3 = _to_lpm_from_mdot(mf["act_pos_in"])
        f4 = _to_lpm_from_mdot(mf["act_pos_out"])
        f5 = _to_lpm_from_mdot(mf["act_neg_in"])
        f6 = _to_lpm_from_mdot(mf["act_neg_out"])

        sim_time.append(obs_time)
        sim_press_pos.append(float(obs[1]))
        sim_press_neg.append(float(obs[2]))
        sim_act_pos.append(float(obs[3]))
        sim_act_neg.append(float(obs[4]))
        for i in range(6):
            sim_ctrl[i].append(float(u[i]))

        if has_real_flow:
            used_real_flow3.append(float(real_flow3_full[idx]))
            used_real_flow4.append(float(real_flow4_full[idx]))
            used_real_flow5.append(float(real_flow5_full[idx]))
            used_real_flow6.append(float(real_flow6_full[idx]))

        calc_flow[0].append(f1)
        calc_flow[1].append(f2)
        calc_flow[2].append(f3)
        calc_flow[3].append(f4)
        calc_flow[4].append(f5)
        calc_flow[5].append(f6)

        net_pos_lpm = _to_lpm_from_mdot(net["act_pos_net_in"])
        net_neg_lpm = _to_lpm_from_mdot(net["act_neg_net_in"])
        calc_net_pos.append(net_pos_lpm)
        calc_net_neg.append(net_neg_lpm)
        mf_act_net_pos.append(net_pos_lpm)
        mf_act_net_neg.append(net_neg_lpm)

        replay_time += dt

    out = {
        "curr_time": np.asarray(sim_time),
        "press_pos": np.asarray(sim_press_pos),
        "press_neg": np.asarray(sim_press_neg),
        "act_pos_press": np.asarray(sim_act_pos),
        "act_neg_press": np.asarray(sim_act_neg),
        "sim_press_pos": np.asarray(sim_press_pos),
        "sim_press_neg": np.asarray(sim_press_neg),
        "sim_act_pos_press": np.asarray(sim_act_pos),
        "sim_act_neg_press": np.asarray(sim_act_neg),
        "ctrl1": np.asarray(sim_ctrl[0]),
        "ctrl2": np.asarray(sim_ctrl[1]),
        "ctrl3": np.asarray(sim_ctrl[2]),
        "ctrl4": np.asarray(sim_ctrl[3]),
        "ctrl5": np.asarray(sim_ctrl[4]),
        "ctrl6": np.asarray(sim_ctrl[5]),
        "calc_flow1": np.asarray(calc_flow[0]),
        "calc_flow2": np.asarray(calc_flow[1]),
        "calc_flow3": np.asarray(calc_flow[2]),
        "calc_flow4": np.asarray(calc_flow[3]),
        "calc_flow5": np.asarray(calc_flow[4]),
        "calc_flow6": np.asarray(calc_flow[5]),
        "calc_net_pos": np.asarray(calc_net_pos),
        "calc_net_neg": np.asarray(calc_net_neg),
        "mf_act_net_pos": np.asarray(mf_act_net_pos),
        "mf_act_net_neg": np.asarray(mf_act_net_neg),
    }

    if has_real_flow:
        real_flow3 = np.asarray(used_real_flow3, dtype=np.float64)
        real_flow4 = np.asarray(used_real_flow4, dtype=np.float64)
        real_flow5 = np.asarray(used_real_flow5, dtype=np.float64)
        real_flow6 = np.asarray(used_real_flow6, dtype=np.float64)
        real_net_pos = real_flow3 - real_flow4
        real_net_neg = real_flow5 - real_flow6
        out.update(
            {
                "real_flow3": real_flow3,
                "real_flow4": real_flow4,
                "real_flow5": real_flow5,
                "real_flow6": real_flow6,
                "real_net_pos": real_net_pos,
                "real_net_neg": real_net_neg,
            }
        )

    return pd.DataFrame(out), dt, has_real_flow


def _print_netflow_quick_metrics(out_df: pd.DataFrame, *, has_real_flow: bool) -> None:
    if not has_real_flow:
        return
    n = len(out_df)
    mae_pos, rmse_pos = _mae_rmse(
        out_df["real_net_pos"].to_numpy(dtype=np.float64),
        out_df["calc_net_pos"].to_numpy(dtype=np.float64),
    )
    mae_neg, rmse_neg = _mae_rmse(
        out_df["real_net_neg"].to_numpy(dtype=np.float64),
        out_df["calc_net_neg"].to_numpy(dtype=np.float64),
    )
    print("[INFO] Net-flow quick metrics (raw index-aligned):")
    print(f"  - net_pos(flow3-flow4): MAE={mae_pos:.4g} RMSE={rmse_pos:.4g} (n={n})")
    print(f"  - net_neg(flow5-flow6): MAE={mae_neg:.4g} RMSE={rmse_neg:.4g} (n={n})")


def _compare_two_replays(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> None:
    cols = [
        "press_pos",
        "press_neg",
        "act_pos_press",
        "act_neg_press",
        "calc_net_pos",
        "calc_net_neg",
        "calc_flow1",
        "calc_flow2",
        "calc_flow3",
        "calc_flow4",
        "calc_flow5",
        "calc_flow6",
    ]
    valid_cols = [c for c in cols if c in df_a.columns and c in df_b.columns]
    if not valid_cols:
        print("[WARN] No comparable columns for replay-to-replay diff.")
        return

    n = min(len(df_a), len(df_b))
    if n == 0:
        print("[WARN] Empty replay output. Skip replay-to-replay diff.")
        return

    print(f"[INFO] Replay equivalence check: {label_a} vs {label_b} (n={n})")
    for col in valid_cols:
        a = df_a[col].to_numpy(dtype=np.float64)[:n]
        b = df_b[col].to_numpy(dtype=np.float64)[:n]
        e = a - b
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e * e)))
        max_abs = float(np.max(np.abs(e)))
        print(f"  - {col}: MAE={mae:.6g} RMSE={rmse:.6g} MAX={max_abs:.6g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="real csv path (e.g. exp/sliced_output.csv)")
    ap.add_argument("--out", default=None, help="output simulation csv path")
    ap.add_argument("--freq", type=float, default=50.0, help="simulation frequency [Hz]")
    ap.add_argument("--delay", type=float, default=0.1, help="simulation observation delay [sec]")
    ap.add_argument("--scale", action="store_true", help="pass through scale flag into PneuSim")
    ap.add_argument("--clip-start-sec", type=float, default=None)
    ap.add_argument("--clip-end-sec", type=float, default=None)
    ap.add_argument("--tail-sec", type=float, default=None)
    ap.add_argument("--no-rebase", action="store_true")
    ap.add_argument("--sim-log", action="store_true", help="enable lib9 C++ runtime logging")
    ap.add_argument("--ctrl-domain", choices=["unit", "bipolar"], default="unit")
    ap.add_argument(
        "--ctrl-smooth-window",
        type=int,
        default=DEFAULT_CTRL_SMOOTH_WINDOW,
        help="pre-smoothing window on replay control (python side, u_eff-domain)",
    )
    ap.add_argument(
        "--sim-ctrl-smooth-window",
        type=int,
        default=1,
        help="internal lib9 smoothing window (C++ side)",
    )
    ap.add_argument(
        "--run-both-smoothing-modes",
        action="store_true",
        help=(
            "run both modes for equivalence check: "
            "(pre=window,sim=0) and (pre=0,sim=window)"
        ),
    )
    ap.add_argument("--debug-input-check", action="store_true")
    args = ap.parse_args()

    df = _load_real_csv(
        args.real,
        clip_start_sec=args.clip_start_sec,
        clip_end_sec=args.clip_end_sec,
        clip_tail_sec=args.tail_sec,
        rebase_time=not args.no_rebase,
    )
    _require_columns(df, REQUIRED_INPUT_COLUMNS)
    traj_time = df["curr_time"].to_numpy(dtype=np.float64)

    ctrl_unit = np.column_stack(
        [
            _convert_ctrl_to_unit(_col(df, "ctrl1"), args.ctrl_domain),
            _convert_ctrl_to_unit(_col(df, "ctrl2"), args.ctrl_domain),
            _convert_ctrl_to_unit(_col(df, "ctrl3"), args.ctrl_domain),
            _convert_ctrl_to_unit(_col(df, "ctrl4"), args.ctrl_domain),
            _convert_ctrl_to_unit(_col(df, "ctrl5"), args.ctrl_domain),
            _convert_ctrl_to_unit(_col(df, "ctrl6"), args.ctrl_domain),
        ]
    )

    base_out_path = _resolve_default_out_path(args.real, args.out)

    if args.run_both_smoothing_modes:
        test_window = max(0, int(args.ctrl_smooth_window))
        mode_a = dict(label=f"pre{test_window}_sim0", pre_window=test_window, sim_window=0)
        mode_b = dict(label=f"pre0_sim{test_window}", pre_window=0, sim_window=test_window)

        print(
            "[INFO] Running both smoothing modes for equivalence check:\n"
            f"  - A: pre={mode_a['pre_window']}, sim={mode_a['sim_window']}\n"
            f"  - B: pre={mode_b['pre_window']}, sim={mode_b['sim_window']}"
        )

        replay_ctrl_a = _build_replay_ctrl(ctrl_unit, pre_smooth_window=mode_a["pre_window"])
        out_df_a, dt_a, has_real_flow_a = _run_single_replay(
            df=df,
            traj_time=traj_time,
            replay_ctrl=replay_ctrl_a,
            freq=float(args.freq),
            delay=float(args.delay),
            scale=bool(args.scale),
            sim_log=bool(args.sim_log),
            sim_ctrl_smooth_window=int(mode_a["sim_window"]),
            no_rebase=bool(args.no_rebase),
            debug_input_check=bool(args.debug_input_check),
            debug_prefix=mode_a["label"],
        )
        out_path_a = _mode_out_path(base_out_path, mode_a["pre_window"], mode_a["sim_window"])
        out_df_a.to_csv(out_path_a, index=False)
        print(f"[INFO] Saved sim9 CSV ({mode_a['label']}): {out_path_a}")
        print(
            "[INFO] Replay finished with fixed input-time stepping "
            f"(dt={dt_a:.6f}s, delay={float(args.delay):.6f}s)"
        )
        _print_netflow_quick_metrics(out_df_a, has_real_flow=has_real_flow_a)

        replay_ctrl_b = _build_replay_ctrl(ctrl_unit, pre_smooth_window=mode_b["pre_window"])
        out_df_b, dt_b, has_real_flow_b = _run_single_replay(
            df=df,
            traj_time=traj_time,
            replay_ctrl=replay_ctrl_b,
            freq=float(args.freq),
            delay=float(args.delay),
            scale=bool(args.scale),
            sim_log=bool(args.sim_log),
            sim_ctrl_smooth_window=int(mode_b["sim_window"]),
            no_rebase=bool(args.no_rebase),
            debug_input_check=bool(args.debug_input_check),
            debug_prefix=mode_b["label"],
        )
        out_path_b = _mode_out_path(base_out_path, mode_b["pre_window"], mode_b["sim_window"])
        out_df_b.to_csv(out_path_b, index=False)
        print(f"[INFO] Saved sim9 CSV ({mode_b['label']}): {out_path_b}")
        print(
            "[INFO] Replay finished with fixed input-time stepping "
            f"(dt={dt_b:.6f}s, delay={float(args.delay):.6f}s)"
        )
        _print_netflow_quick_metrics(out_df_b, has_real_flow=has_real_flow_b)

        _compare_two_replays(
            out_df_a,
            out_df_b,
            label_a=mode_a["label"],
            label_b=mode_b["label"],
        )
        return

    replay_ctrl = _build_replay_ctrl(ctrl_unit, pre_smooth_window=int(args.ctrl_smooth_window))
    out_df, dt, has_real_flow = _run_single_replay(
        df=df,
        traj_time=traj_time,
        replay_ctrl=replay_ctrl,
        freq=float(args.freq),
        delay=float(args.delay),
        scale=bool(args.scale),
        sim_log=bool(args.sim_log),
        sim_ctrl_smooth_window=int(args.sim_ctrl_smooth_window),
        no_rebase=bool(args.no_rebase),
        debug_input_check=bool(args.debug_input_check),
    )

    out_df.to_csv(base_out_path, index=False)
    print(f"[INFO] Saved sim9 CSV: {base_out_path}")
    print(
        "[INFO] Replay finished with fixed input-time stepping "
        f"(dt={dt:.6f}s, delay={float(args.delay):.6f}s)"
    )
    _print_netflow_quick_metrics(out_df, has_real_flow=has_real_flow)


if __name__ == "__main__":
    main()
