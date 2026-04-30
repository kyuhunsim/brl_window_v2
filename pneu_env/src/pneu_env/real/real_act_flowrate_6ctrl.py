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

import os
import json
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from pneu_utils.utils import get_pkg_path


ATM = 101.325


# ==============================
# Manual runtime config
# Edit this block directly.
# ==============================
CTRL_MODE = "suite"    # "random" | "const" | "suite"
CONST_CTRLS = [0.85, 1.0, 1.0, 1.0, 1.0, 1.0]

# Only the first two channels are actively excited by default:
#   ctrl1: pos_ctrl
#   ctrl2: neg_ctrl
# The other actuator channels are held fixed unless you edit FIXED_TAIL_CTRLS.
ACTIVE_CTRL_COUNT = 2
FIXED_TAIL_CTRLS = [0.0, 0.0, 0.0, 0.0]

# Random mode params
RAND_HOLD_MIN = 1      # sec (inclusive)
RAND_HOLD_MAX = 5     # sec (inclusive)
RAND_MIN = 0.85
RAND_MAX = 1.0

# Suite mode runs several profiles in one recording.
# Durations are intentionally editable here instead of CLI arguments.
SUITE_PROFILES = [
    dict(
        name="random_2ctrl",
        mode="random",
        duration=300.0,
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
        ctrl1_period=10.0,
        ctrl1_phase=0.0,
        ctrl2_offset=0.925,
        ctrl2_amp=0.075,
        ctrl2_period=10.0,
        ctrl2_phase=np.pi,
    ),
    dict(
        name="bangbang_pos_only",
        mode="bangbang",
        duration=120.0,
        periods=[10.0, 5.0, 3.0, 2.0, 1.0],
        ctrl1=dict(mode="bangbang", min=0.85, max=1.0, phase="normal"),
        ctrl2=dict(mode="fixed", fixed=1.0),
    ),
    dict(
        name="bangbang_neg_only",
        mode="bangbang",
        duration=120.0,
        periods=[10.0, 5.0, 3.0, 2.0, 1.0],
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
]

PROFILE_MODE_IDS = dict(
    idle=0,
    random=1,
    step=2,
    bangbang=3,
    const=4,
    sin=5,
)


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


def main():
    # -----------------------------
    # 설정 (필요 시 여기만 수정)
    # -----------------------------
    freq = 50.0                 # control freq [Hz]
    duration = 1000.0            # experiment duration [sec], ignored by suite mode
    tag = ""                    # filename tag suffix

    ctrl_mode = CTRL_MODE
    rand_hold_min = int(RAND_HOLD_MIN)   # hold time range [sec] (inclusive)
    rand_hold_max = int(RAND_HOLD_MAX)
    rand_min = float(RAND_MIN)
    rand_max = float(RAND_MAX)

    if ctrl_mode not in ("random", "const", "suite"):
        raise ValueError(f"CTRL_MODE must be random|const|suite, got: {ctrl_mode}")
    if rand_hold_min < 1 or rand_hold_max < rand_hold_min:
        raise ValueError("RAND_HOLD_MIN/MAX must satisfy 1 <= min <= max")
    if not (0.0 <= rand_min <= 1.0 and 0.0 <= rand_max <= 1.0):
        raise ValueError("RAND_MIN/RAND_MAX must be in [0,1]")
    if rand_min > rand_max:
        raise ValueError("RAND_MIN must be <= RAND_MAX")

    if ctrl_mode == "suite":
        duration = sum(float(profile["duration"]) for profile in SUITE_PROFILES)

    tag = f"_{tag}" if tag else ""
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    save_file_name = f"{formatted_time}_Flowrate_RND6_{ctrl_mode}{tag}"

    print(
        f"[INFO] mode={ctrl_mode}, duration={duration:.1f}s, "
        f"active_ctrl_count={ACTIVE_CTRL_COUNT}, fixed_tail={FIXED_TAIL_CTRLS}, "
        f"rand_hold=[{rand_hold_min},{rand_hold_max}]s, rand_range=[{rand_min},{rand_max}]"
    )
    period = 1.0 / max(freq, 1e-9)
    print(f"[INFO] target control loop: {freq:.1f}Hz ({period * 1000.0:.3f}ms)")

    unit_ctrls = np.zeros(6, dtype=np.float64)
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
    )

    try:
        script_start_time = time.time()
        curr_time = float(obs_state["time"])
        prev_time = None
        started = False
        suite_start_time = 0.0
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
                if ctrl_mode == "suite":
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
                unit_ctrls = np.zeros(6, dtype=np.float64)

            ctrl_unit = np.asarray(unit_ctrls, dtype=np.float64)
            act_unit = ctrl_unit[2:]
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
            if obs_time <= prev_obs_time + 1e-9:
                stale_obs_count += 1

            # obs_act.json이 이전 run의 time을 들고 있을 수 있음.
            # 우선순위:
            # 1) time 감소(리셋) 감지
            # 2) reset 이벤트가 없더라도, fresh run(0~2s 구간의 증가 time) 감지 시 시작
            if prev_time is None:
                prev_time = obs_time
                if 0.0 <= obs_time <= 1.0:
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

                if obs_time >= float(duration):
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
        print(f"[INFO] Saved experiment CSV: {out_path}")


if __name__ == "__main__":
    main()
