#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_act_flowrate_6ctrl.py

- JSON 브리지를 직접 구동하는 6채널 실험 러너.
- 6개 제어를 하나의 모드로 다룸: random / const / suite.
- tcpip 브리지(tcpip_connect_act*.py)와 함께 쓰면 obs_act.json에
  flowrate1~flowrate6가 들어오고, 이를 exp CSV로 저장한다.

출력 CSV 컬럼은 tuner가 바로 먹을 수 있게 맞춤:
  curr_time, ctrl_pos, ctrl_neg, press_pos, press_neg, flowrate1~flowrate6, ...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
from pathlib import Path
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

# Support direct execution from the checkout as well as editable installs.
_ROOT = Path(__file__).resolve().parents[4]
for _package in ("pneu_env", "pneu_utils"):
    sys.path.insert(0, str(_ROOT / _package / "src"))

from pneu_env.real.flowrate_profiles import make_stair_levels, stair_value
from pneu_env.real.joint_tuning_profiles import JointTuningSuite
from pneu_utils.utils import get_pkg_path


ATM = 101.325


# ==============================
# Manual runtime config
# Edit this block directly.
# ==============================
CTRL_MODE = "suite"    # "random" | "const" | "suite"
CONST_CTRLS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
RUN_MODE = "joint_tuning"  # six-channel 657 s suite; other names below select legacy 2-channel suites

# Saved joint settings: pneu_env/config/collection/lib3_joint_tuning_type{1,2}.json
# type1: primary suite; type2: additional ramps on idle channels.
# Both bound scheduled supply u < 0.85 visits to 0.2 s (not a pressure interlock).
COLLECTION_TYPE = "type2"

# Legacy modes only (joint_tuning always generates all six channels):
#   ctrl1: pos_ctrl
#   ctrl2: neg_ctrl
# The other actuator channels are held fixed unless you edit FIXED_TAIL_CTRLS.
ACTIVE_CTRL_COUNT = 2
FIXED_TAIL_CTRLS = [0.0, 0.0, 0.0, 0.0]

# Legacy random mode params (joint_tuning reads the selected saved JSON preset).
RAND_HOLD_MIN = 0.5      # sec (inclusive)
RAND_HOLD_MAX = 3     # sec (inclusive)
RAND_MIN = 0.8
RAND_MAX = 1.0

# Suite mode runs several profiles in one recording.
# Durations are intentionally editable here instead of CLI arguments.
SUITE_PROFILES = [
    dict(
        name="random_2ctrl",
        mode="random",
        duration=200.0,
        hold_min=1,
        hold_max=5,
        min=0.85,
        max=1.0,
    ),
    dict(
        name="step_2ctrl",
        mode="step",
        duration=180.0,
        time_step=3.0,
        ctrl1_values=[0.85, 1.0, 0.9, 0.98, 0.87, 0.95],
        ctrl2_values=[1.0, 0.85, 0.95, 0.88, 0.98, 0.9],
    ),
    dict(
        name="sin_2ctrl",
        mode="sin",
        duration=180.0,
        ctrl1_offset=0.925,
        ctrl1_amp=0.075,
        ctrl1_period=5.0,
        ctrl1_phase=0.0,
        ctrl2_offset=0.925,
        ctrl2_amp=0.075,
        ctrl2_period=5.0,
        ctrl2_phase=np.pi,
    ),
    dict(
        name="bangbang_pos_only",
        mode="bangbang",
        duration=120.0,
        periods=[5.0, 3.0, 2.0, 1.0, 0.5],
        ctrl1=dict(mode="bangbang", min=0.85, max=1.0, phase="normal"),
        ctrl2=dict(mode="fixed", fixed=1.0),
    ),
    dict(
        name="bangbang_neg_only",
        mode="bangbang",
        duration=120.0,
        periods=[5.0, 3.0, 2.0, 1.0, 0.5],
        ctrl1=dict(mode="fixed", fixed=0.85),
        ctrl2=dict(mode="bangbang", min=0.85, max=1.0, phase="normal"),
    ),
    dict(
        name="bangbang_both_inverse",
        mode="bangbang",
        duration=120.0,
        periods=[10.0, 5.0, 3.0, 2.0, 1.0],
        ctrl1=dict(mode="bangbang", min=0.85, max=1.0, phase="normal"),
        ctrl2=dict(mode="bangbang", min=0.85, max=1.0, phase="inverse"),
    ),
    dict(
        name="grid_hold_pos_2ctrl",
        mode="grid_hold",
        duration=88.0,
        ctrl1_levels=[0.90, 0.925, 0.95, 0.975, 1.00, 0.975, 0.95, 0.925],
        ctrl2_levels=[1.00],
        hold_sec=10.0,
        transition_sec=1.0,
    ),
    dict(
        name="grid_hold_neg_2ctrl",
        mode="grid_hold",
        duration=88.0,
        ctrl1_levels=[1.00],
        ctrl2_levels=[0.90, 0.925, 0.95, 0.975, 1.00, 0.975, 0.95, 0.925],
        hold_sec=10.0,
        transition_sec=1.0,
    ),
    dict(
        name="slow_ramp_pos_2ctrl",
        mode="ramp",
        duration=140.0,
        ctrl1_start=0.90,
        ctrl1_peak=1.00,
        ctrl1_end=0.90,
        ctrl2_start=1.00,
        ctrl2_peak=1.00,
        ctrl2_end=1.00,
        ramp_up_sec=60.0,
        hold_peak_sec=10.0,
        ramp_down_sec=60.0,
        hold_end_sec=10.0,
    ),
    dict(
        name="slow_ramp_neg_2ctrl",
        mode="ramp",
        duration=140.0,
        ctrl1_start=1.00,
        ctrl1_peak=1.00,
        ctrl1_end=1.00,
        ctrl2_start=0.90,
        ctrl2_peak=1.00,
        ctrl2_end=0.90,
        ramp_up_sec=60.0,
        hold_peak_sec=10.0,
        ramp_down_sec=60.0,
        hold_end_sec=10.0,
    ),
    dict(
        name="step_response_pos_2ctrl",
        mode="step_response",
        duration=56.0,
        ctrl1_low=0.90,
        ctrl1_high=1.00,
        ctrl2_low=1.00,
        ctrl2_high=1.00,
        settle_sec=8.0,
        high_sec=6.0,
        cycles=4,
    ),
    dict(
        name="step_response_neg_2ctrl",
        mode="step_response",
        duration=56.0,
        ctrl1_low=1.00,
        ctrl1_high=1.00,
        ctrl2_low=0.90,
        ctrl2_high=1.00,
        settle_sec=8.0,
        high_sec=6.0,
        cycles=4,
    ),
    dict(
        name="settle_all_open_a",
        mode="hold",
        duration=8.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="triangle_pos_0p1hz",
        mode="triangle_wave",
        duration=40.0,
        period=10.0,
        ctrl1_min=0.85,
        ctrl1_max=1.00,
        ctrl2_min=1.00,
        ctrl2_max=1.00,
    ),
    dict(
        name="triangle_pos_0p5hz",
        mode="triangle_wave",
        duration=16.0,
        period=2.0,
        ctrl1_min=0.85,
        ctrl1_max=1.00,
        ctrl2_min=1.00,
        ctrl2_max=1.00,
    ),
    dict(
        name="triangle_pos_1hz",
        mode="triangle_wave",
        duration=10.0,
        period=1.0,
        ctrl1_min=0.85,
        ctrl1_max=1.00,
        ctrl2_min=1.00,
        ctrl2_max=1.00,
    ),
    # dict(
    #     name="triangle_pos_2hz",
    #     mode="triangle_wave",
    #     duration=6.0,
    #     period=0.5,
    #     ctrl1_min=0.85,
    #     ctrl1_max=1.00,
    #     ctrl2_min=1.00,
    #     ctrl2_max=1.00,
    # ),
    dict(
        name="settle_all_open_b",
        mode="hold",
        duration=8.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="triangle_neg_0p1hz",
        mode="triangle_wave",
        duration=40.0,
        period=10.0,
        ctrl1_min=1.00,
        ctrl1_max=1.00,
        ctrl2_min=0.85,
        ctrl2_max=1.00,
    ),
    dict(
        name="triangle_neg_0p5hz",
        mode="triangle_wave",
        duration=16.0,
        period=2.0,
        ctrl1_min=1.00,
        ctrl1_max=1.00,
        ctrl2_min=0.85,
        ctrl2_max=1.00,
    ),
    dict(
        name="triangle_neg_1hz",
        mode="triangle_wave",
        duration=10.0,
        period=1.0,
        ctrl1_min=1.00,
        ctrl1_max=1.00,
        ctrl2_min=0.85,
        ctrl2_max=1.00,
    ),
    # dict(
    #     name="triangle_neg_2hz",
    #     mode="triangle_wave",
    #     duration=6.0,
    #     period=0.5,
    #     ctrl1_min=1.00,
    #     ctrl1_max=1.00,
    #     ctrl2_min=0.85,
    #     ctrl2_max=1.00,
    # ),
    dict(
        name="settle_all_open_c",
        mode="hold",
        duration=8.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="random_stair_pos",
        mode="random_staircase",
        duration=79.75,
        target="ctrl1",
        low=0.85,
        high=1.00,
        n_random_steps=9,
        hold_sec=4.0,
        transition_sec=0.25,
        seed=101,
        fixed_other=1.0,
    ),
    dict(
        name="settle_all_open_d",
        mode="hold",
        duration=8.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="random_stair_neg",
        mode="random_staircase",
        duration=79.75,
        target="ctrl2",
        low=0.85,
        high=1.00,
        n_random_steps=9,
        hold_sec=4.0,
        transition_sec=0.25,
        seed=202,
        fixed_other=1.0,
    ),
    dict(
        name="settle_all_open_e",
        mode="hold",
        duration=8.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="pump_all_open_start",
        mode="hold",
        duration=12.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="pump_close_pulse_a",
        mode="hold",
        duration=3.0,
        ctrl1=0.0,
        ctrl2=0.0,
    ),
    dict(
        name="pump_pos_sweep_a",
        mode="ramp",
        duration=30.0,
        ctrl1_start=0.85,
        ctrl1_peak=1.00,
        ctrl1_end=0.85,
        ctrl2_start=1.00,
        ctrl2_peak=1.00,
        ctrl2_end=1.00,
        ramp_up_sec=12.0,
        hold_peak_sec=3.0,
        ramp_down_sec=12.0,
        hold_end_sec=3.0,
    ),
    dict(
        name="pump_all_open_mid_a",
        mode="hold",
        duration=5.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="pump_neg_sweep_a",
        mode="ramp",
        duration=30.0,
        ctrl1_start=1.00,
        ctrl1_peak=1.00,
        ctrl1_end=1.00,
        ctrl2_start=0.85,
        ctrl2_peak=1.00,
        ctrl2_end=0.85,
        ramp_up_sec=12.0,
        hold_peak_sec=3.0,
        ramp_down_sec=12.0,
        hold_end_sec=3.0,
    ),
    dict(
        name="pump_all_open_mid_b",
        mode="hold",
        duration=5.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="pump_close_pulse_b",
        mode="hold",
        duration=3.0,
        ctrl1=0.0,
        ctrl2=0.0,
    ),
    dict(
        name="pump_pos_sweep_b",
        mode="ramp",
        duration=30.0,
        ctrl1_start=0.85,
        ctrl1_peak=1.00,
        ctrl1_end=0.85,
        ctrl2_start=1.00,
        ctrl2_peak=1.00,
        ctrl2_end=1.00,
        ramp_up_sec=12.0,
        hold_peak_sec=3.0,
        ramp_down_sec=12.0,
        hold_end_sec=3.0,
    ),
    dict(
        name="pump_all_open_mid_c",
        mode="hold",
        duration=5.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
    dict(
        name="pump_neg_sweep_b",
        mode="ramp",
        duration=30.0,
        ctrl1_start=1.00,
        ctrl1_peak=1.00,
        ctrl1_end=1.00,
        ctrl2_start=0.85,
        ctrl2_peak=1.00,
        ctrl2_end=0.85,
        ramp_up_sec=12.0,
        hold_peak_sec=3.0,
        ramp_down_sec=12.0,
        hold_end_sec=3.0,
    ),
    dict(
        name="pump_all_open_end",
        mode="hold",
        duration=8.0,
        ctrl1=1.0,
        ctrl2=1.0,
    ),
]

RUN_MODE_PROFILE_NAMES = dict(
    all=None,
    static=("grid_hold_pos_2ctrl", "grid_hold_neg_2ctrl"),
    hysteresis=("slow_ramp_pos_2ctrl", "slow_ramp_neg_2ctrl"),
    dynamic=("step_response_pos_2ctrl", "step_response_neg_2ctrl"),
    hysteresis_combo=(
        "settle_all_open_a",
        "triangle_pos_0p1hz",
        "triangle_pos_0p5hz",
        "triangle_pos_1hz",
        "triangle_pos_2hz",
        "settle_all_open_b",
        "triangle_neg_0p1hz",
        "triangle_neg_0p5hz",
        "triangle_neg_1hz",
        "triangle_neg_2hz",
        "settle_all_open_c",
        "random_stair_pos",
        "settle_all_open_d",
        "random_stair_neg",
        "settle_all_open_e",
    ),
    pump_tuning=(
        "pump_all_open_start",
        "pump_close_pulse_a",
        "pump_pos_sweep_a",
        "pump_all_open_mid_a",
        "pump_neg_sweep_a",
        "pump_all_open_mid_b",
        "pump_close_pulse_b",
        "pump_pos_sweep_b",
        "pump_all_open_mid_c",
        "pump_neg_sweep_b",
        "pump_all_open_end",
    ),
)

PROFILE_MODE_IDS = dict(
    idle=0,
    random=1,
    step=2,
    bangbang=3,
    const=4,
    sin=5,
    grid_hold=6,
    ramp=7,
    step_response=8,
    hold=9,
    triangle_wave=10,
    random_staircase=11,
)


def _clip_ctrl(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _get_profile_levels(profile: dict, prefix: str) -> list[float]:
    levels_key = f"{prefix}_levels"
    if levels_key in profile:
        levels = [float(v) for v in profile[levels_key]]
        if not levels:
            raise ValueError(f"{levels_key} must not be empty")
        return [_clip_ctrl(v) for v in levels]

    start = float(profile[f"{prefix}_start"])
    peak = float(profile[f"{prefix}_peak"])
    end = float(profile.get(f"{prefix}_end", start))
    delta = float(profile[f"{prefix}_delta"])
    return make_stair_levels(start=start, peak=peak, end=end, delta=delta)


def _ramp_value(
    local_time: float,
    *,
    start: float,
    peak: float,
    end: float,
    ramp_up_sec: float,
    hold_peak_sec: float,
    ramp_down_sec: float,
    hold_end_sec: float,
) -> tuple[float, float]:
    t = float(max(0.0, local_time))
    ramp_up = float(max(0.0, ramp_up_sec))
    peak_hold = float(max(0.0, hold_peak_sec))
    ramp_down = float(max(0.0, ramp_down_sec))
    end_hold = float(max(0.0, hold_end_sec))

    if ramp_up > 0.0 and t < ramp_up:
        alpha = t / ramp_up
        return _clip_ctrl(start + alpha * (peak - start)), ramp_up
    t -= ramp_up

    if t < peak_hold:
        return _clip_ctrl(peak), max(peak_hold, 1e-9)
    t -= peak_hold

    if ramp_down > 0.0 and t < ramp_down:
        alpha = t / ramp_down
        return _clip_ctrl(peak + alpha * (end - peak)), ramp_down
    t -= ramp_down

    if t < end_hold:
        return _clip_ctrl(end), max(end_hold, 1e-9)
    return _clip_ctrl(end), max(end_hold, 1e-9)


def _triangle_value(local_time: float, *, period: float, v_min: float, v_max: float) -> float:
    period = float(max(period, 1e-9))
    phase = (float(local_time) % period) / period
    if phase < 0.5:
        alpha = phase / 0.5
        value = float(v_min) + alpha * (float(v_max) - float(v_min))
    else:
        alpha = (phase - 0.5) / 0.5
        value = float(v_max) + alpha * (float(v_min) - float(v_max))
    return _clip_ctrl(value)


def _random_stair_levels(profile: dict) -> list[float]:
    low = _clip_ctrl(profile["low"])
    high = _clip_ctrl(profile["high"])
    n_random_steps = int(profile.get("n_random_steps", 4))
    if n_random_steps < 0:
        raise ValueError("random_staircase n_random_steps must be >= 0")

    seed = int(profile.get("seed", 0))
    rng = np.random.default_rng(seed)
    mids = []
    if n_random_steps > 0:
        mids = sorted(float(x) for x in rng.uniform(low, high, size=n_random_steps))
    up = [low] + mids + [high]
    down = list(reversed(up[:-1]))
    return [_clip_ctrl(v) for v in (up + down)]


def _resolve_tcpip_dir() -> str:
    candidates: list[str] = []

    env_dir = os.getenv("PNEU_TCPIP_DIR", "").strip()
    if env_dir:
        candidates.append(os.path.abspath(env_dir))

    # Source-tree default (this script lives in .../src/pneu_env/real).
    candidates.append(
        os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tcpip")
        )
    )

    try:
        pkg_root = get_pkg_path("pneu_env")
        candidates.append(os.path.join(pkg_root, "src", "pneu_env", "tcpip"))
        candidates.append(os.path.join(pkg_root, "tcpip"))
    except Exception:
        pass

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        norm = os.path.abspath(path)
        if norm not in seen:
            unique_candidates.append(norm)
            seen.add(norm)

    for tcpip_dir in unique_candidates:
        if os.path.isfile(os.path.join(tcpip_dir, "tcpip_connect_act.py")):
            return tcpip_dir
    for tcpip_dir in unique_candidates:
        if os.path.isdir(tcpip_dir):
            return tcpip_dir

    return unique_candidates[0]


def _initial_obs_state() -> dict[str, float]:
    return dict(
        time=0.0,
        pos_press=101.325,
        neg_press=101.325,
        pos_ref=101.325,
        neg_ref=101.325,
        pos_ctrl=1.0,
        neg_ctrl=1.0,
        act_pos_press=101.325,
        act_neg_press=101.325,
        act_pos_ref=0.0,
        act_neg_ref=0.0,
        act_pos_ctrl1=0.0,
        act_pos_ctrl2=0.0,
        act_neg_ctrl1=0.0,
        act_neg_ctrl2=0.0,
        angle=0.0,
        angle_reference=0.0,
        angular_vel=0.0,
        len1=float("nan"),
        vel1=float("nan"),
        flowrate1=0.0,
        flowrate2=0.0,
        flowrate3=0.0,
        flowrate4=0.0,
        flowrate5=0.0,
        flowrate6=0.0,
    )


def _write_json_atomic(path: str, payload: dict[str, float]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _tail_ctrls() -> np.ndarray:
    tail = np.asarray(FIXED_TAIL_CTRLS, dtype=np.float64)
    expected = 6 - int(ACTIVE_CTRL_COUNT)
    if tail.shape != (expected,):
        raise ValueError(f"FIXED_TAIL_CTRLS must be shape ({expected},), got {tail.shape}")
    return np.clip(tail, 0.0, 1.0)


def _compose_6ctrl(active_ctrls: np.ndarray) -> np.ndarray:
    active_ctrls = np.asarray(active_ctrls, dtype=np.float64)
    if active_ctrls.shape != (int(ACTIVE_CTRL_COUNT),):
        raise ValueError(
            f"active_ctrls must be shape ({ACTIVE_CTRL_COUNT},), got {active_ctrls.shape}"
        )
    return np.clip(np.concatenate([active_ctrls, _tail_ctrls()]), 0.0, 1.0)


def _suite_profile_at(suite_time: float):
    elapsed = 0.0
    for idx, profile in enumerate(SUITE_PROFILES):
        profile_duration = float(profile["duration"])
        if suite_time < elapsed + profile_duration:
            return idx, profile, suite_time - elapsed
        elapsed += profile_duration
    return len(SUITE_PROFILES), None, 0.0


def _select_suite_profiles(run_mode: str) -> list[dict]:
    mode = str(run_mode).strip().lower()
    if not mode or mode == "all":
        return list(SUITE_PROFILES)

    selected_names = RUN_MODE_PROFILE_NAMES.get(mode, None)
    if selected_names is None:
        selected_names = (run_mode,)

    selected = [profile for profile in SUITE_PROFILES if profile.get("name") in selected_names]
    if not selected:
        available = ", ".join(profile.get("name", "<unnamed>") for profile in SUITE_PROFILES)
        raise ValueError(
            f"RUN_MODE={run_mode!r} did not match any suite profile. Available: {available}"
        )
    return selected


def _bangbang_channel(local_time: float, period: float, cfg: dict) -> float:
    mode = cfg.get("mode", "fixed")
    if mode == "fixed":
        return float(cfg["fixed"])
    if mode != "bangbang":
        raise ValueError(f"Unknown bangbang channel mode: {mode}")

    first_half = (local_time % period) < (0.5 * period)
    phase = cfg.get("phase", "normal")
    if phase == "inverse":
        first_half = not first_half
    elif phase != "normal":
        raise ValueError(f"Unknown bangbang phase: {phase}")

    return float(cfg["min"] if first_half else cfg["max"])


def _bangbang_period_at(local_time: float, profile: dict) -> tuple[float, float]:
    periods = [float(p) for p in profile["periods"]]
    cycles_per_period = float(profile.get("cycles_per_period", 2.0))
    schedule_duration = sum(period * cycles_per_period for period in periods)
    if schedule_duration <= 0.0:
        raise ValueError("bangbang schedule duration must be positive")

    t = local_time % schedule_duration
    elapsed = 0.0
    for period in periods:
        segment_duration = period * cycles_per_period
        if t < elapsed + segment_duration:
            return period, t - elapsed
        elapsed += segment_duration
    return periods[-1], 0.0


def _make_suite_ctrls(
    *,
    suite_time: float,
    current_ctrls: np.ndarray,
    next_change_time: float,
) -> tuple[np.ndarray, float, int, int, float, float]:
    profile_idx, profile, local_time = _suite_profile_at(suite_time)
    if profile is None:
        return _compose_6ctrl(np.array([1.0, 1.0], dtype=np.float64)), next_change_time, profile_idx, 0, local_time, 0.0

    mode = profile["mode"]
    mode_id = int(PROFILE_MODE_IDS[mode])
    period = 0.0

    if mode == "random":
        if local_time >= next_change_time:
            active_ctrls = np.random.uniform(
                low=float(profile["min"]),
                high=float(profile["max"]),
                size=int(ACTIVE_CTRL_COUNT),
            ).astype(np.float64)
            hold_sec = float(np.random.randint(int(profile["hold_min"]), int(profile["hold_max"]) + 1))
            next_change_time = local_time + hold_sec
            return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period
        return current_ctrls.copy(), next_change_time, profile_idx, mode_id, local_time, period

    if mode == "step":
        time_step = float(profile["time_step"])
        if time_step <= 0.0:
            raise ValueError("step time_step must be positive")
        ctrl1_values = profile["ctrl1_values"]
        ctrl2_values = profile["ctrl2_values"]
        if len(ctrl1_values) != len(ctrl2_values):
            raise ValueError("ctrl1_values and ctrl2_values must have the same length")
        idx = int(local_time // time_step) % len(ctrl1_values)
        active_ctrls = np.array([ctrl1_values[idx], ctrl2_values[idx]], dtype=np.float64)
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, time_step

    if mode == "sin":
        ctrl1_period = float(profile["ctrl1_period"])
        ctrl2_period = float(profile["ctrl2_period"])
        if ctrl1_period <= 0.0 or ctrl2_period <= 0.0:
            raise ValueError("sin periods must be positive")
        ctrl1 = float(profile["ctrl1_offset"]) + float(profile["ctrl1_amp"]) * np.sin(
            2.0 * np.pi * local_time / ctrl1_period + float(profile["ctrl1_phase"])
        )
        ctrl2 = float(profile["ctrl2_offset"]) + float(profile["ctrl2_amp"]) * np.sin(
            2.0 * np.pi * local_time / ctrl2_period + float(profile["ctrl2_phase"])
        )
        period = min(ctrl1_period, ctrl2_period)
        active_ctrls = np.array([ctrl1, ctrl2], dtype=np.float64)
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period

    if mode == "grid_hold":
        hold_sec = float(profile["hold_sec"])
        transition_sec = float(profile.get("transition_sec", 0.0))
        if hold_sec <= 0.0:
            raise ValueError("grid_hold hold_sec must be positive")
        if transition_sec < 0.0:
            raise ValueError("grid_hold transition_sec must be >= 0")

        ctrl1_levels = _get_profile_levels(profile, "ctrl1")
        ctrl2_levels = _get_profile_levels(profile, "ctrl2")
        ctrl1 = stair_value(
            local_time,
            levels=ctrl1_levels,
            hold_s=hold_sec,
            transition_s=transition_sec,
        )
        ctrl2 = stair_value(
            local_time,
            levels=ctrl2_levels,
            hold_s=hold_sec,
            transition_s=transition_sec,
        )
        period = hold_sec + transition_sec
        active_ctrls = np.array([ctrl1, ctrl2], dtype=np.float64)
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period

    if mode == "hold":
        active_ctrls = np.array(
            [float(profile["ctrl1"]), float(profile["ctrl2"])],
            dtype=np.float64,
        )
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, float(profile["duration"])

    if mode == "triangle_wave":
        period = float(profile["period"])
        if period <= 0.0:
            raise ValueError("triangle_wave period must be positive")
        ctrl1 = _triangle_value(
            local_time,
            period=period,
            v_min=float(profile["ctrl1_min"]),
            v_max=float(profile["ctrl1_max"]),
        )
        ctrl2 = _triangle_value(
            local_time,
            period=period,
            v_min=float(profile["ctrl2_min"]),
            v_max=float(profile["ctrl2_max"]),
        )
        active_ctrls = np.array([ctrl1, ctrl2], dtype=np.float64)
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period

    if mode == "random_staircase":
        hold_sec = float(profile["hold_sec"])
        transition_sec = float(profile.get("transition_sec", 0.0))
        if hold_sec <= 0.0:
            raise ValueError("random_staircase hold_sec must be positive")
        if transition_sec < 0.0:
            raise ValueError("random_staircase transition_sec must be >= 0")
        levels = _random_stair_levels(profile)
        moving_ctrl = stair_value(
            local_time,
            levels=levels,
            hold_s=hold_sec,
            transition_s=transition_sec,
        )
        fixed_other = _clip_ctrl(profile.get("fixed_other", 1.0))
        target = str(profile.get("target", "ctrl1")).lower()
        if target == "ctrl1":
            active_ctrls = np.array([moving_ctrl, fixed_other], dtype=np.float64)
        elif target == "ctrl2":
            active_ctrls = np.array([fixed_other, moving_ctrl], dtype=np.float64)
        else:
            raise ValueError(f"random_staircase target must be ctrl1 or ctrl2, got: {target}")
        period = hold_sec + transition_sec
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period

    if mode == "ramp":
        ramp_up_sec = float(profile["ramp_up_sec"])
        hold_peak_sec = float(profile.get("hold_peak_sec", 0.0))
        ramp_down_sec = float(profile["ramp_down_sec"])
        hold_end_sec = float(profile.get("hold_end_sec", 0.0))

        ctrl1, _ = _ramp_value(
            local_time,
            start=float(profile["ctrl1_start"]),
            peak=float(profile["ctrl1_peak"]),
            end=float(profile.get("ctrl1_end", profile["ctrl1_start"])),
            ramp_up_sec=ramp_up_sec,
            hold_peak_sec=hold_peak_sec,
            ramp_down_sec=ramp_down_sec,
            hold_end_sec=hold_end_sec,
        )
        ctrl2, _ = _ramp_value(
            local_time,
            start=float(profile["ctrl2_start"]),
            peak=float(profile["ctrl2_peak"]),
            end=float(profile.get("ctrl2_end", profile["ctrl2_start"])),
            ramp_up_sec=ramp_up_sec,
            hold_peak_sec=hold_peak_sec,
            ramp_down_sec=ramp_down_sec,
            hold_end_sec=hold_end_sec,
        )
        period = max(ramp_up_sec + hold_peak_sec + ramp_down_sec + hold_end_sec, 1e-9)
        active_ctrls = np.array([ctrl1, ctrl2], dtype=np.float64)
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period

    if mode == "step_response":
        settle_sec = float(profile["settle_sec"])
        high_sec = float(profile["high_sec"])
        cycles = int(profile.get("cycles", 1))
        if settle_sec <= 0.0 or high_sec <= 0.0:
            raise ValueError("step_response settle_sec/high_sec must be positive")
        if cycles <= 0:
            raise ValueError("step_response cycles must be >= 1")

        cycle_period = settle_sec + high_sec
        cycle_idx = min(int(local_time // cycle_period), cycles - 1)
        cycle_time = local_time - cycle_idx * cycle_period
        high = cycle_time >= settle_sec

        ctrl1 = float(profile["ctrl1_high"] if high else profile["ctrl1_low"])
        ctrl2 = float(profile["ctrl2_high"] if high else profile["ctrl2_low"])
        active_ctrls = np.array([ctrl1, ctrl2], dtype=np.float64)
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, cycle_period

    if mode == "bangbang":
        period, period_time = _bangbang_period_at(local_time, profile)
        active_ctrls = np.array(
            [
                _bangbang_channel(period_time, period, profile["ctrl1"]),
                _bangbang_channel(period_time, period, profile["ctrl2"]),
            ],
            dtype=np.float64,
        )
        return _compose_6ctrl(active_ctrls), next_change_time, profile_idx, mode_id, local_time, period

    raise ValueError(f"Unknown suite profile mode: {mode}")


def _make_unit_ctrls(
    ctrl_mode: str,
    *,
    curr_time: float,
    rand_min: float,
    rand_max: float,
    rand_hold_min: int,
    rand_hold_max: int,
    next_change_time: float,
    current_ctrls: np.ndarray,
    const_ctrls: np.ndarray,
) -> tuple[np.ndarray, float]:
    if ctrl_mode == "random":
        if curr_time >= next_change_time:
            active_ctrls = np.random.uniform(
                low=rand_min,
                high=rand_max,
                size=int(ACTIVE_CTRL_COUNT),
            ).astype(np.float64)
            ctrl_unit = np.random.uniform(
                low=rand_min,
                high=rand_max,
                size=6,
            ).astype(np.float64)
            ctrl_unit = _compose_6ctrl(active_ctrls)
            hold_sec = float(np.random.randint(rand_hold_min, rand_hold_max + 1))
            next_change_time = curr_time + hold_sec
            return np.clip(ctrl_unit, 0.0, 1.0), next_change_time
        return current_ctrls.copy(), next_change_time
    if ctrl_mode == "const":
        return np.clip(const_ctrls.astype(np.float64), 0.0, 1.0), next_change_time
    raise ValueError(f"CTRL_MODE must be random|const|suite, got: {ctrl_mode}")


def _build_ctrl_payload(
    *,
    obs_state: dict[str, float],
    goal: np.ndarray,
    ctrl_unit: np.ndarray,
    act_unit: np.ndarray,
    start_time: float,
) -> dict[str, float]:
    payload = dict(
        time=float(time.time() - start_time),
        pos_press=float(obs_state["pos_press"]),
        neg_press=float(obs_state["neg_press"]),
        pos_ref=float(goal[0]),
        neg_ref=float(goal[1]),
        pos_ctrl=float(ctrl_unit[0]),
        neg_ctrl=float(ctrl_unit[1]),
        act_pos_press=float(obs_state["act_pos_press"]),
        act_neg_press=float(obs_state["act_neg_press"]),
        act_pos_ref=float(goal[0]),
        act_neg_ref=float(goal[1]),
        act_pos_ctrl1=float(act_unit[0]),
        act_pos_ctrl2=float(act_unit[1]),
        act_neg_ctrl1=float(act_unit[2]),
        act_neg_ctrl2=float(act_unit[3]),
        angle=float(obs_state["angle"]),
        angle_reference=float(obs_state["angle_reference"]),
        angular_vel=float(obs_state["angular_vel"]),
        flowrate1=float(obs_state["flowrate1"]),
        flowrate2=float(obs_state["flowrate2"]),
        flowrate3=float(obs_state["flowrate3"]),
        flowrate4=float(obs_state["flowrate4"]),
        flowrate5=float(obs_state["flowrate5"]),
        flowrate6=float(obs_state["flowrate6"]),
    )
    return payload


def _read_obs_state(
    *,
    obs_json_path: str,
    prev_state: dict[str, float],
    sen_period: float,
    max_wait_s: float | None = None,
) -> dict[str, float]:
    prev_time = float(prev_state["time"])
    wait_budget = max_wait_s if max_wait_s is not None else max(0.03, 3.0 * sen_period)
    wait_budget = max(0.0, float(wait_budget))
    deadline = time.perf_counter() + wait_budget

    def _get_float(data: dict, key: str, default: float) -> float:
        value = data.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_any_float(data: dict, keys: tuple[str, ...], default: float) -> float:
        for key in keys:
            if key in data:
                return _get_float(data, key, default)
        return default

    while True:
        try:
            with open(obs_json_path, "r", encoding="utf-8") as f:
                obs = json.load(f)
            obs_time = _get_float(obs, "time", prev_state["time"])
            now = time.perf_counter()
            if obs_time <= prev_time + 1e-9 and now < deadline:
                remain = max(0.0, deadline - now)
                time.sleep(min(0.001, 0.2 * sen_period, remain))
                continue

            state = dict(prev_state)
            state["time"] = obs_time
            state["pos_press"] = _get_float(obs, "pos_press", state["pos_press"])
            state["neg_press"] = _get_float(obs, "neg_press", state["neg_press"])
            state["pos_ref"] = _get_float(obs, "pos_ref", state["pos_ref"])
            state["neg_ref"] = _get_float(obs, "neg_ref", state["neg_ref"])
            state["pos_ctrl"] = _get_float(obs, "pos_ctrl", state["pos_ctrl"])
            state["neg_ctrl"] = _get_float(obs, "neg_ctrl", state["neg_ctrl"])
            state["act_pos_press"] = _get_float(obs, "act_pos_press", state["act_pos_press"])
            state["act_neg_press"] = _get_float(obs, "act_neg_press", state["act_neg_press"])
            state["act_pos_ref"] = _get_float(obs, "act_pos_ref", state["act_pos_ref"])
            state["act_neg_ref"] = _get_float(obs, "act_neg_ref", state["act_neg_ref"])
            state["act_pos_ctrl1"] = _get_float(obs, "act_pos_ctrl1", state["act_pos_ctrl1"])
            state["act_pos_ctrl2"] = _get_float(obs, "act_pos_ctrl2", state["act_pos_ctrl2"])
            state["act_neg_ctrl1"] = _get_float(obs, "act_neg_ctrl1", state["act_neg_ctrl1"])
            state["act_neg_ctrl2"] = _get_float(obs, "act_neg_ctrl2", state["act_neg_ctrl2"])
            state["angle"] = _get_float(obs, "angle", state["angle"])
            state["angle_reference"] = _get_float(obs, "angle_reference", state["angle_reference"])
            state["angular_vel"] = _get_float(obs, "angular_vel", state["angular_vel"])
            state["len1"] = _get_any_float(
                obs,
                ("len1", "length", "length_m", "disp", "displacement"),
                state["len1"],
            )
            state["vel1"] = _get_any_float(
                obs,
                ("vel1", "velocity", "vel_ms", "length_velocity"),
                state["vel1"],
            )
            state["flowrate1"] = _get_float(obs, "flowrate1", state["flowrate1"])
            state["flowrate2"] = _get_float(obs, "flowrate2", state["flowrate2"])
            state["flowrate3"] = _get_float(obs, "flowrate3", state["flowrate3"])
            state["flowrate4"] = _get_float(obs, "flowrate4", state["flowrate4"])
            state["flowrate5"] = _get_float(obs, "flowrate5", state["flowrate5"])
            state["flowrate6"] = _get_float(obs, "flowrate6", state["flowrate6"])
            return state
        except FileNotFoundError:
            if time.perf_counter() >= deadline:
                return dict(prev_state)
            time.sleep(min(0.001, 0.2 * sen_period))
        except json.JSONDecodeError:
            if time.perf_counter() >= deadline:
                return dict(prev_state)
            time.sleep(min(0.001, 0.2 * sen_period))
        except Exception:
            if time.perf_counter() >= deadline:
                return dict(prev_state)
            time.sleep(min(0.001, 0.2 * sen_period))


def main(argv=None):
    # -----------------------------
    # 설정 (필요 시 여기만 수정)
    # -----------------------------
    freq = 50.0                 # control freq [Hz]
    duration = 300.0            # experiment duration [sec], ignored by suite mode
    tag = ""                    # filename tag suffix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode", default=RUN_MODE, help="joint_tuning or a legacy suite name")
    parser.add_argument("--type", dest="collection_type", choices=("type1", "type2"),
                        default=COLLECTION_TYPE, help="saved joint suite preset (default: type2)")
    parser.add_argument("--preview", type=Path, metavar="DIR",
                        help="export the joint suite CSV/config/plot without bridge I/O")
    args = parser.parse_args(argv)

    ctrl_mode = CTRL_MODE
    run_mode = str(args.run_mode).strip()
    rand_hold_min = float(RAND_HOLD_MIN)   # hold time range [sec] (inclusive)
    rand_hold_max = float(RAND_HOLD_MAX)
    rand_min = float(RAND_MIN)
    rand_max = float(RAND_MAX)
    suite_profiles = list(SUITE_PROFILES)
    joint_suite = None
    observation_timeout_sec = None

    if ctrl_mode not in ("random", "const", "suite"):
        raise ValueError(f"CTRL_MODE must be random|const|suite, got: {ctrl_mode}")
    # if rand_hold_min < 1 or rand_hold_max < rand_hold_min:
    #     raise ValueError("RAND_HOLD_MIN/MAX must satisfy 1 <= min <= max")
    if not (0.0 <= rand_min <= 1.0 and 0.0 <= rand_max <= 1.0):
        raise ValueError("RAND_MIN/RAND_MAX must be in [0,1]")
    if rand_min > rand_max:
        raise ValueError("RAND_MIN must be <= RAND_MAX")

    if ctrl_mode == "suite":
        if run_mode == "joint_tuning":
            joint_suite = JointTuningSuite.from_preset(args.collection_type)
            config = joint_suite.config
            freq, rand_min, rand_max = config.freq, config.command_min, config.command_max
            observation_timeout_sec = joint_suite.observation_timeout_sec
            suite_profiles = joint_suite.profiles
            tag = (f"{joint_suite.collection_type}_" +
                   (tag or f"u{rand_min:.2f}_supply{config.supply_low_max_sec:.2f}s"))
            if config.idle_ramps:
                tag += "_idle_ramps"
        else:
            suite_profiles = _select_suite_profiles(run_mode)
        duration = sum(float(profile["duration"]) for profile in suite_profiles)
        globals()["SUITE_PROFILES"] = suite_profiles

    if args.preview:
        if joint_suite is None:
            parser.error("--preview requires CTRL_MODE='suite' and --run-mode joint_tuning")
        output = joint_suite.write_preview(args.preview)
        print(f"[PREVIEW] {joint_suite.collection_type}: {duration:.1f}s, "
              f"{len(joint_suite.commands)} commands: {output}")
        print(f"[PREVIEW] maximum continuous u < {joint_suite.config.supply_low_threshold}: "
              f"{joint_suite.max_low_seconds()} seconds (valves 1..6)")
        print("[PREVIEW] full-open percentage (valves 1..6): "
              + str([round(r["full_open_percent"], 2) for r in joint_suite.coverage()]))
        return

    tag = f"_{tag}" if tag else ""
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    run_mode_suffix = f"_{run_mode}" if ctrl_mode == "suite" and run_mode else ""
    save_file_name = f"{formatted_time}_Flowrate_RND6_{ctrl_mode}{run_mode_suffix}{tag}"

    print(
        f"[INFO] mode={ctrl_mode}, duration={duration:.1f}s, "
        f"active_ctrl_count={6 if joint_suite else ACTIVE_CTRL_COUNT}, "
        f"fixed_tail={[] if joint_suite else FIXED_TAIL_CTRLS}, "
        f"rand_hold={[1, 3] if joint_suite else [rand_hold_min, rand_hold_max]}s, "
        f"rand_range=[{rand_min},{rand_max}], "
        f"run_mode={run_mode}"
    )
    if ctrl_mode == "suite":
        print("[INFO] selected suite profiles: " + ", ".join(profile["name"] for profile in suite_profiles))
    period = 1.0 / max(freq, 1e-9)
    print(f"[INFO] target control loop: {freq:.1f}Hz ({period * 1000.0:.3f}ms)")
    if joint_suite:
        print(f"[INFO] {joint_suite.collection_type}, preset={joint_suite.saved_preset['path']}")
        print(f"[INFO] supply valves 1/2: u < {joint_suite.config.supply_low_threshold} command budget "
              f"{joint_suite.config.supply_low_max_sec}s; not a pressure interlock")

    unit_ctrls = np.ones(6, dtype=np.float64) if joint_suite else np.zeros(6, dtype=np.float64)
    requested_ctrls = unit_ctrls.copy()
    low_limited = np.zeros(2, dtype=bool)
    idle_ramp_mask = np.zeros(6, dtype=bool)
    const_ctrls = np.asarray(CONST_CTRLS, dtype=np.float64)
    if const_ctrls.shape != (6,):
        raise ValueError(f"CONST_CTRLS must be shape (6,), got {const_ctrls.shape}")
    const_ctrls = np.clip(const_ctrls, 0.0, 1.0)
    next_change_time = 0.0

    tcpip_dir = _resolve_tcpip_dir()
    ctrl_json_path = os.path.join(tcpip_dir, "ctrl_act.json")
    obs_json_path = os.path.join(tcpip_dir, "obs_act.json")
    print(f"[INFO] tcpip dir: {tcpip_dir}")
    obs_state = _initial_obs_state()
    goal = np.array([ATM, 0.0], dtype=np.float64)

    data = dict(
        time=deque(),
        press_pos=deque(),
        press_neg=deque(),
        act_pos_press=deque(),
        act_neg_press=deque(),
        ctrl1=deque(),
        ctrl2=deque(),
        ctrl3=deque(),
        ctrl4=deque(),
        ctrl5=deque(),
        ctrl6=deque(),
        flow1=deque(),
        flow2=deque(),
        flow3=deque(),
        flow4=deque(),
        flow5=deque(),
        flow6=deque(),
        anlge=deque(),
        angle_vel=deque(),
        profile_idx=deque(),
        profile_mode=deque(),
        profile_time=deque(),
        profile_period=deque(),
        suite_time=deque(),
        command_time=deque(),
        supply_low_limited1=deque(),
        supply_low_limited2=deque(),
    )
    for channel in range(1, 7):
        data[f"command_ctrl{channel}"] = deque()
        data[f"requested_ctrl{channel}"] = deque()
        data[f"idle_ramp{channel}"] = deque()

    try:
        script_start_time = time.time()
        script_start_tick = time.perf_counter()
        curr_time = float(obs_state["time"])
        prev_time = None
        started = False
        suite_start_time = 0.0
        suite_start_tick = None
        suite_elapsed = 0.0
        last_fresh_tick = time.perf_counter()
        profile_idx = -1
        profile_mode = 0
        profile_time = 0.0
        profile_period = 0.0
        wait_print_next_wall = 0.0
        next_tick = time.perf_counter()
        last_tick = None
        timing_window = max(int(freq), 1)
        loop_dt_ms_hist: deque[float] = deque(maxlen=timing_window)
        obs_wait_ms_hist: deque[float] = deque(maxlen=timing_window)
        stale_obs_count = 0

        _write_json_atomic(
            ctrl_json_path,
            _build_ctrl_payload(
                obs_state=obs_state,
                goal=goal,
                ctrl_unit=unit_ctrls[:2],
                act_unit=unit_ctrls[2:],
                start_time=script_start_time,
            ),
        )

        while True:
            now_tick = time.perf_counter()
            if now_tick < next_tick:
                time.sleep(next_tick - now_tick)
                now_tick = time.perf_counter()
            elif now_tick - next_tick > 2.0 * period:
                # If we are too late, resync to avoid drift accumulation.
                next_tick = now_tick
            next_tick += period

            if last_tick is not None:
                loop_dt_ms_hist.append((now_tick - last_tick) * 1000.0)
            last_tick = now_tick

            if started:
                if joint_suite:
                    # Do not stretch a low-command visit when sensor time stalls.
                    suite_elapsed = max(0.0, time.perf_counter() - suite_start_tick)
                    unit_ctrls, metadata = joint_suite.sample(suite_elapsed)
                    requested_ctrls = metadata["requested"]
                    low_limited = metadata["low_limited"]
                    idle_ramp_mask = metadata["idle_ramp_mask"]
                    profile_idx = metadata["profile_idx"]
                    profile_mode = PROFILE_MODE_IDS[metadata["profile_mode"]]
                    profile_time = metadata["profile_time"]
                    profile_period = metadata["profile_period"]
                elif ctrl_mode == "suite":
                    suite_time = max(0.0, curr_time - suite_start_time)
                    (
                        unit_ctrls,
                        next_change_time,
                        profile_idx,
                        profile_mode,
                        profile_time,
                        profile_period,
                    ) = _make_suite_ctrls(
                        suite_time=suite_time,
                        current_ctrls=unit_ctrls,
                        next_change_time=next_change_time,
                    )
                else:
                    unit_ctrls, next_change_time = _make_unit_ctrls(
                        ctrl_mode,
                        curr_time=curr_time,
                        rand_min=rand_min,
                        rand_max=rand_max,
                        rand_hold_min=rand_hold_min,
                        rand_hold_max=rand_hold_max,
                        next_change_time=next_change_time,
                        current_ctrls=unit_ctrls,
                        const_ctrls=const_ctrls,
                    )
                    profile_idx = 0
                    profile_mode = int(PROFILE_MODE_IDS[ctrl_mode])
                    profile_time = curr_time
                    profile_period = 0.0
            else:
                unit_ctrls = np.ones(6, dtype=np.float64) if joint_suite else np.zeros(6, dtype=np.float64)

            if joint_suite is None:
                requested_ctrls = unit_ctrls.copy()

            ctrl_unit = np.asarray(unit_ctrls, dtype=np.float64)
            act_unit = ctrl_unit[2:]
            command_tick = time.perf_counter()
            _write_json_atomic(
                ctrl_json_path,
                _build_ctrl_payload(
                    obs_state=obs_state,
                    goal=goal,
                    ctrl_unit=ctrl_unit[:2],
                    act_unit=act_unit,
                    start_time=script_start_time,
                ),
            )

            prev_obs_time = float(obs_state["time"])
            obs_read_start = time.perf_counter()
            obs_state = _read_obs_state(
                obs_json_path=obs_json_path,
                prev_state=obs_state,
                sen_period=1.0 / max(freq, 1e-6),
                max_wait_s=0.6 * period,
            )
            obs_wait_ms_hist.append((time.perf_counter() - obs_read_start) * 1000.0)
            obs = obs_state.copy()
            info = {"Observation": obs}
            o = info["Observation"]
            obs_time = float(o["time"])
            observation_fresh = (obs_time > prev_obs_time + 1e-9
                                 or obs_time < prev_obs_time - 1e-3)
            if observation_fresh:
                last_fresh_tick = time.perf_counter()
            if joint_suite and time.perf_counter() - last_fresh_tick > observation_timeout_sec:
                raise TimeoutError("No fresh observation; ending collection and publishing all-open")
            if obs_time <= prev_obs_time + 1e-9:
                stale_obs_count += 1

            # obs_act.json이 이전 run의 time을 들고 있을 수 있음.
            # 우선순위:
            # 1) time 감소(리셋) 감지
            # 2) reset 이벤트가 없더라도, fresh run(0~2s 구간의 증가 time) 감지 시 시작
            if prev_time is None:
                prev_time = obs_time
                if 0.0 <= obs_time <= 1.0 and (joint_suite is None or observation_fresh):
                    started = True
                    for k in data.keys():
                        data[k].clear()
                    next_change_time = obs_time
                    suite_start_time = obs_time
                    print(f"[INFO] fresh start detected (time={obs_time:.3f}), start logging.")
            else:
                if not started:
                    if obs_time < prev_time - 1e-3:
                        started = True
                        for k in data.keys():
                            data[k].clear()
                        # time reset -> restart random schedule
                        next_change_time = obs_time
                        suite_start_time = obs_time
                        print(f"[INFO] time reset detected ({prev_time:.3f} -> {obs_time:.3f}), start logging.")
                    elif (
                        0.0 <= prev_time <= 1.0
                        and obs_time > prev_time + 1e-6
                        and obs_time <= 2.0
                    ):
                        started = True
                        for k in data.keys():
                            data[k].clear()
                        next_change_time = obs_time
                        suite_start_time = obs_time
                        print(
                            f"[INFO] monotonic fresh time detected "
                            f"({prev_time:.3f} -> {obs_time:.3f}), start logging."
                        )

            prev_time = obs_time
            if started and suite_start_tick is None:
                suite_start_tick = time.perf_counter()

            if started:
                fr1 = float(o["flowrate1"])
                fr2 = float(o["flowrate2"])
                fr3 = float(o["flowrate3"])
                fr4 = float(o["flowrate4"])
                fr5 = float(o["flowrate5"])
                fr6 = float(o["flowrate6"])

                data["time"].append(obs_time)
                data["press_pos"].append(float(o["pos_press"]))
                data["press_neg"].append(float(o["neg_press"]))
                data["act_pos_press"].append(float(o["act_pos_press"]))
                data["act_neg_press"].append(float(o["act_neg_press"]))
                data["ctrl1"].append(float(o["pos_ctrl"]))
                data["ctrl2"].append(float(o["neg_ctrl"]))
                data["ctrl3"].append(float(o["act_pos_ctrl1"]))
                data["ctrl4"].append(float(o["act_pos_ctrl2"]))
                data["ctrl5"].append(float(o["act_neg_ctrl1"]))
                data["ctrl6"].append(float(o["act_neg_ctrl2"]))
                data["flow1"].append(fr1)
                data["flow2"].append(fr2)
                data["flow3"].append(fr3)
                data["flow4"].append(fr4)
                data["flow5"].append(fr5)
                data["flow6"].append(fr6)
                data["anlge"].append(float(o["angle"]))
                data["angle_vel"].append(float(o["angular_vel"]))
                data["profile_idx"].append(float(profile_idx))
                data["profile_mode"].append(float(profile_mode))
                data["profile_time"].append(float(profile_time))
                data["profile_period"].append(float(profile_period))
                data["suite_time"].append(suite_elapsed)
                data["command_time"].append(command_tick - script_start_tick)
                data["supply_low_limited1"].append(float(low_limited[0]))
                data["supply_low_limited2"].append(float(low_limited[1]))
                for channel in range(6):
                    data[f"command_ctrl{channel + 1}"].append(float(ctrl_unit[channel]))
                    data[f"requested_ctrl{channel + 1}"].append(float(requested_ctrls[channel]))
                    data[f"idle_ramp{channel + 1}"].append(float(idle_ramp_mask[channel]))

                if len(data["time"]) % int(freq) == 0:
                    loop_dt_ms = float(np.mean(loop_dt_ms_hist)) if loop_dt_ms_hist else float("nan")
                    loop_hz = (1000.0 / loop_dt_ms) if loop_dt_ms > 0.0 else float("nan")
                    obs_wait_ms = float(np.mean(obs_wait_ms_hist)) if obs_wait_ms_hist else float("nan")
                    print(
                        f"[INFO] t={obs_time:.2f}s "
                        f"profile=({profile_idx},{profile_mode}) "
                        f"ctrl=({o['pos_ctrl']:.3f},{o['neg_ctrl']:.3f},"
                        f"{o['act_pos_ctrl1']:.3f},{o['act_pos_ctrl2']:.3f},"
                        f"{o['act_neg_ctrl1']:.3f},{o['act_neg_ctrl2']:.3f}) "
                        f"P=({o['pos_press']:.1f},{o['neg_press']:.1f},"
                        f"{o['act_pos_press']:.1f},{o['act_neg_press']:.1f}) "
                        f"FR=({fr1:.3f},{fr2:.3f},{fr3:.3f},{fr4:.3f},{fr5:.3f},{fr6:.3f}) "
                        f"| loop={loop_hz:.1f}Hz dt={loop_dt_ms:.3f}ms "
                        f"obs_wait={obs_wait_ms:.3f}ms stale={stale_obs_count}"
                    )
                    stale_obs_count = 0

                finished = suite_elapsed >= duration if joint_suite else obs_time >= float(duration)
                if finished:
                    break
            else:
                now_wall = time.time()
                if now_wall >= wait_print_next_wall:
                    loop_dt_ms = float(np.mean(loop_dt_ms_hist)) if loop_dt_ms_hist else float("nan")
                    loop_hz = (1000.0 / loop_dt_ms) if loop_dt_ms > 0.0 else float("nan")
                    obs_wait_ms = float(np.mean(obs_wait_ms_hist)) if obs_wait_ms_hist else float("nan")
                    print(
                        f"[WAIT] waiting for fresh start signal "
                        f"(obs_time={obs_time:.3f}, prev_time={prev_time:.3f}) "
                        f"| loop={loop_hz:.1f}Hz dt={loop_dt_ms:.3f}ms "
                        f"obs_wait={obs_wait_ms:.3f}ms stale={stale_obs_count}"
                    )
                    stale_obs_count = 0
                    wait_print_next_wall = now_wall + 1.0

            curr_time = float(o["time"])

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping experiment")

    finally:
        # 안전하게 밸브를 열어 놓고 종료
        try:
            safe_ctrls = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            _write_json_atomic(
                ctrl_json_path,
                _build_ctrl_payload(
                    obs_state=obs_state,
                    goal=goal,
                    ctrl_unit=safe_ctrls[:2],
                    act_unit=safe_ctrls[2:],
                    start_time=script_start_time,
                ),
            )
            time.sleep(0.1)
        except Exception:
            pass

        for k, v in data.items():
            data[k] = np.array(v, dtype=np.float64)

        df = pd.DataFrame(data)
        exp_dir = os.path.join(get_pkg_path("pneu_env"), "exp")
        os.makedirs(exp_dir, exist_ok=True)
        out_path = os.path.join(exp_dir, f"{save_file_name}.csv")
        df.to_csv(out_path, index=False)
        if joint_suite:
            manifest = joint_suite.manifest()
            manifest.update(csv_path=out_path, logged_rows=len(df),
                clock="monotonic elapsed time since fresh-start detection",
                ctrl_columns="ctrl1..6 are observation echoes; command_ctrl1..6 are published commands",
                requested_columns="requested_ctrl1..6 are planned commands before the supply limiter",
                observation_timeout_sec=observation_timeout_sec)
            for source in (Path(__file__), Path(sys.modules[JointTuningSuite.__module__].__file__)):
                manifest.setdefault("source_sha256", {})[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            Path(out_path).with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[INFO] Saved experiment CSV: {out_path}")


if __name__ == "__main__":
    main()
