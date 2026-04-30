#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_sim3_from_real_csv.py

Generate a lib3(sim3) replay CSV from a real CSV and include net-flow columns.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sim3 import PneuSim

STD_RHO = 1.20411831637462
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


def _unit_to_sim3_ctrl(ctrl_unit: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * np.asarray(ctrl_unit, dtype=np.float64) - 1.0, -1.0, 1.0)


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
        path = f"{base}_sim3.csv"
    if os.path.splitext(path)[1] == "":
        path = f"{path}.csv"
    return path


def _has_real_flow(df: pd.DataFrame) -> bool:
    return all(
        (f"flow{i}" in df.columns) or (f"flowrate{i}" in df.columns)
        for i in range(3, 7)
    )


def _real_flow_col(df: pd.DataFrame, idx: int) -> np.ndarray:
    return _col(df, f"flow{idx}", f"flowrate{idx}")


def _sim3_flow_lpm(sim: PneuSim) -> tuple[float, float, float, float, float, float]:
    mf = sim.get_mass_flowrate()
    if len(mf) < 10:
        raise ValueError(f"sim3 mass-flow vector must have at least 10 values, got {len(mf)}")

    return (
        _to_lpm_from_mdot(mf[4]),  # chamber_pos_valve
        _to_lpm_from_mdot(mf[5]),  # chamber_neg_valve
        _to_lpm_from_mdot(mf[6]),  # act_pos_in
        _to_lpm_from_mdot(mf[7]),  # act_pos_out
        _to_lpm_from_mdot(mf[8]),  # act_neg_in
        _to_lpm_from_mdot(mf[9]),  # act_neg_out
    )


def _run_single_replay(
    *,
    df: pd.DataFrame,
    traj_time: np.ndarray,
    ctrl_unit: np.ndarray,
    freq: float,
    delay: float,
    scale: bool,
    sim_log: bool,
    no_rebase: bool,
    debug_input_check: bool,
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

    has_real_flow = _has_real_flow(df)
    real_flow3_full = _real_flow_col(df, 3) if has_real_flow else None
    real_flow4_full = _real_flow_col(df, 4) if has_real_flow else None
    real_flow5_full = _real_flow_col(df, 5) if has_real_flow else None
    real_flow6_full = _real_flow_col(df, 6) if has_real_flow else None
    used_real_flow3 = []
    used_real_flow4 = []
    used_real_flow5 = []
    used_real_flow6 = []

    while replay_time < t_end + 1e-12:
        u_unit, idx = _row_at_time(traj_time, ctrl_unit, replay_time, idx)
        u_sim = _unit_to_sim3_ctrl(u_unit)
        if debug_input_check and dbg_count < 15:
            print(
                f"[DBG] replay_time={replay_time:.6f} src_idx={idx} "
                f"src_csv_time={traj_time[idx]:.6f} "
                f"u_unit={np.array2string(u_unit, precision=4, suppress_small=False)} "
                f"u_sim3={np.array2string(u_sim, precision=4, suppress_small=False)}"
            )
            dbg_count += 1

        obs, _ = sim.observe(u_sim)
        obs_time = float(obs[0]) + time_offset
        f1, f2, f3, f4, f5, f6 = _sim3_flow_lpm(sim)

        sim_time.append(obs_time)
        sim_press_pos.append(float(obs[1]))
        sim_press_neg.append(float(obs[2]))
        sim_act_pos.append(float(obs[3]))
        sim_act_neg.append(float(obs[4]))
        for i in range(6):
            sim_ctrl[i].append(float(u_unit[i]))

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

        net_pos_lpm = f3 - f4
        net_neg_lpm = f5 - f6
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="real csv path (e.g. exp/sliced_output.csv)")
    ap.add_argument("--out", default=None, help="output simulation csv path")
    ap.add_argument("--freq", type=float, default=50.0, help="simulation frequency [Hz]")
    ap.add_argument("--delay", type=float, default=0.1, help="simulation observation delay [sec]")
    ap.add_argument(
        "--scale",
        action="store_true",
        help="use sim3 compressed action scaling [-1,1] -> [0.7,1.0]",
    )
    ap.add_argument("--clip-start-sec", type=float, default=None)
    ap.add_argument("--clip-end-sec", type=float, default=None)
    ap.add_argument("--tail-sec", type=float, default=None)
    ap.add_argument("--no-rebase", action="store_true")
    ap.add_argument("--sim-log", action="store_true", help="enable lib3 C++ runtime logging")
    ap.add_argument("--ctrl-domain", choices=["unit", "bipolar"], default="unit")
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
    print("[INFO] Using raw control replay (lib3 smoothing is fixed at 1)")
    out_df, dt, has_real_flow = _run_single_replay(
        df=df,
        traj_time=traj_time,
        ctrl_unit=ctrl_unit,
        freq=float(args.freq),
        delay=float(args.delay),
        scale=bool(args.scale),
        sim_log=bool(args.sim_log),
        no_rebase=bool(args.no_rebase),
        debug_input_check=bool(args.debug_input_check),
    )

    out_df.to_csv(base_out_path, index=False)
    print(f"[INFO] Saved sim3 CSV: {base_out_path}")
    print(
        "[INFO] Replay finished with fixed input-time stepping "
        f"(dt={dt:.6f}s, delay={float(args.delay):.6f}s)"
    )
    _print_netflow_quick_metrics(out_df, has_real_flow=has_real_flow)


if __name__ == "__main__":
    main()
