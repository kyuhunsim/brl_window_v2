#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_act_flowrate_6ctrl.py

- JSON 브리지를 직접 구동하는 6채널 실험 러너.
- 6개 제어를 하나의 모드로 다룸: random / const.
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

from utils.utils import get_pkg_path

try:
    from .flowrate_profiles import read_flowrate_from_obs_json
except ImportError:
    from flowrate_profiles import read_flowrate_from_obs_json


ATM = 101.325


# ==============================
# Manual runtime config
# Edit this block directly.
# ==============================
CTRL_MODE = "random"    # "random" | "const"
CONST_CTRLS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Random mode params
RAND_HOLD_MIN = 1      # sec (inclusive)
RAND_HOLD_MAX = 10     # sec (inclusive)
RAND_MIN = 0.8
RAND_MAX = 1.0


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
            ctrl_unit = np.random.uniform(
                low=rand_min,
                high=rand_max,
                size=6,
            ).astype(np.float64)
            hold_sec = float(np.random.randint(rand_hold_min, rand_hold_max + 1))
            next_change_time = curr_time + hold_sec
            return np.clip(ctrl_unit, 0.0, 1.0), next_change_time
        return current_ctrls.copy(), next_change_time
    if ctrl_mode == "const":
        return np.clip(const_ctrls.astype(np.float64), 0.0, 1.0), next_change_time
    raise ValueError(f"CTRL_MODE must be random|const, got: {ctrl_mode}")


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
) -> dict[str, float]:
    prev_time = float(prev_state["time"])
    deadline = time.time() + max(0.03, 3.0 * sen_period)

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
            if obs_time <= prev_time + 1e-9 and time.time() < deadline:
                time.sleep(min(0.002, 0.2 * sen_period))
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
            if time.time() >= deadline:
                return dict(prev_state)
        except json.JSONDecodeError:
            if time.time() >= deadline:
                return dict(prev_state)
        except Exception:
            if time.time() >= deadline:
                return dict(prev_state)


def main():
    # -----------------------------
    # 설정 (필요 시 여기만 수정)
    # -----------------------------
    freq = 200.0                 # control freq [Hz]
    duration = 500.0            # experiment duration [sec]
    tag = ""                    # filename tag suffix

    ctrl_mode = CTRL_MODE
    rand_hold_min = int(RAND_HOLD_MIN)   # hold time range [sec] (inclusive)
    rand_hold_max = int(RAND_HOLD_MAX)
    rand_min = float(RAND_MIN)
    rand_max = float(RAND_MAX)

    if ctrl_mode not in ("random", "const"):
        raise ValueError(f"CTRL_MODE must be random|const, got: {ctrl_mode}")
    if rand_hold_min < 1 or rand_hold_max < rand_hold_min:
        raise ValueError("RAND_HOLD_MIN/MAX must satisfy 1 <= min <= max")
    if not (0.0 <= rand_min <= 1.0 and 0.0 <= rand_max <= 1.0):
        raise ValueError("RAND_MIN/RAND_MAX must be in [0,1]")
    if rand_min > rand_max:
        raise ValueError("RAND_MIN must be <= RAND_MAX")

    tag = f"_{tag}" if tag else ""
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    save_file_name = f"{formatted_time}_Flowrate_RND6_{ctrl_mode}{tag}"

    print(
        f"[INFO] mode={ctrl_mode}, rand_hold=[{rand_hold_min},{rand_hold_max}]s, "
        f"rand_range=[{rand_min},{rand_max}]"
    )

    unit_ctrls = np.zeros(6, dtype=np.float64)
    const_ctrls = np.asarray(CONST_CTRLS, dtype=np.float64)
    if const_ctrls.shape != (6,):
        raise ValueError(f"CONST_CTRLS must be shape (6,), got {const_ctrls.shape}")
    const_ctrls = np.clip(const_ctrls, 0.0, 1.0)
    next_change_time = 0.0

    ctrl_json_path = os.path.join(get_pkg_path("pneu_env"), "tcpip/ctrl_act.json")
    obs_json_path = os.path.join(get_pkg_path("pneu_env"), "tcpip/obs_act.json")
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
    )

    try:
        script_start_time = time.time()
        flag_time = time.time()
        curr_time = float(obs_state["time"])
        prev_time = None
        started = False
        wait_print_next_wall = 0.0

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
            if started:
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

            elapsed = time.time() - flag_time
            sleep_time = max((1.0 / freq) - elapsed - 0.004, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)
            flag_time = time.time()

            obs_state = _read_obs_state(
                obs_json_path=obs_json_path,
                prev_state=obs_state,
                sen_period=1.0 / max(freq, 1e-6),
            )
            obs = obs_state.copy()
            info = {"Observation": obs}
            o = info["Observation"]
            obs_time = float(o["time"])

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
                    print(f"[INFO] fresh start detected (time={obs_time:.3f}), start logging.")
            else:
                if not started:
                    if obs_time < prev_time - 1e-3:
                        started = True
                        for k in data.keys():
                            data[k].clear()
                        # time reset -> restart random schedule
                        next_change_time = obs_time
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
                        print(
                            f"[INFO] monotonic fresh time detected "
                            f"({prev_time:.3f} -> {obs_time:.3f}), start logging."
                        )

            prev_time = obs_time

            if started:
                fr1, fr2, fr3, fr4, fr5, fr6 = read_flowrate_from_obs_json(obs_json_path)

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

                if len(data["time"]) % int(freq) == 0:
                    print(
                        f"[INFO] t={obs_time:.2f}s "
                        f"ctrl=({o['pos_ctrl']:.3f},{o['neg_ctrl']:.3f},"
                        f"{o['act_pos_ctrl1']:.3f},{o['act_pos_ctrl2']:.3f},"
                        f"{o['act_neg_ctrl1']:.3f},{o['act_neg_ctrl2']:.3f}) "
                        f"P=({o['pos_press']:.1f},{o['neg_press']:.1f},"
                        f"{o['act_pos_press']:.1f},{o['act_neg_press']:.1f}) "
                        f"FR=({fr1:.3f},{fr2:.3f},{fr3:.3f},{fr4:.3f},{fr5:.3f},{fr6:.3f})"
                    )

                if obs_time >= float(duration):
                    break
            else:
                now_wall = time.time()
                if now_wall >= wait_print_next_wall:
                    print(
                        f"[WAIT] waiting for fresh start signal "
                        f"(obs_time={obs_time:.3f}, prev_time={prev_time:.3f})"
                    )
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
