#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_act_flowrate_6ctrl.py

- real_act(PneuRealAct) 실행 루프에 랜덤 입력을 합친 스크립트(6개 제어).
- 메인 챔버 제어 2개(pos/neg)는 CtrlRandom으로 [-1,1] 랜덤 입력.
- 액추에이터 제어 4개는 zero/const/random/stair 모드로 제어 가능(0~1 범위).
- tcpip 브리지(tcpip_connect_act*.py)와 함께 쓰면 obs_act.json에
  flowrate1~flowrate6가 들어오고, 이를 exp CSV로 저장한다.

출력 CSV 컬럼은 tuner가 바로 먹을 수 있게 맞춤:
  curr_time, ctrl_pos, ctrl_neg, press_pos, press_neg, flowrate1~flowrate6, ...
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


from env.real.real_act import PneuRealAct
from utils.utils import get_pkg_path


ATM = 101.325


# ==============================
# Manual runtime config
# Edit this block directly.
# ==============================
MAIN_MODE = "stair"    # "zero" | "const" | "random" | "stair" (main chamber 2ch)
ACT_MODE = "stair"     # "zero" | "const" | "random" | "stair" (actuator 4ch)
USE_SCALE = False       # True: [0.8,1.0] scaling, False: [0,1]

# Random mode params
RAND_HOLD_MIN = 1      # sec (inclusive)
RAND_HOLD_MAX = 10     # sec (inclusive)
RAND_MIN = 0.8
RAND_MAX = 1.0

# Stair mode params (directional step-up/down with hold)
# List를 직접 쓰지 않고 step으로 자동 생성:
# start -> ... -> peak 까지 STEP_DELTA만큼 증가 후,
# peak -> ... -> end 까지 STEP_DELTA만큼 감소
STAIR_START = 0.75
STAIR_PEAK = 1.00
STAIR_END = 0.70
STAIR_DELTA = 0.02
STAIR_HOLD_SEC = 5.0          # 각 스텝 레벨 유지 시간
STAIR_TRANSITION_SEC = 0.2      # 레벨 간 선형 이동 시간
# Positive phase means that channel runs earlier in time.
MAIN_STAIR_PHASE_SEC = [0.0, 0.0]
ACT_STAIR_PHASE_SEC = [0.0, 0.5, 1.0, 1.5]


def _make_stair_levels(start: float, peak: float, end: float, delta: float) -> list[float]:
    d = float(abs(delta))
    if d <= 0.0:
        raise ValueError("STAIR_DELTA must be > 0")

    lo = float(np.clip(start, 0.0, 1.0))
    hi = float(np.clip(peak, 0.0, 1.0))
    last = float(np.clip(end, 0.0, 1.0))

    if hi < lo:
        lo, hi = hi, lo

    up = [lo]
    v = lo
    while v + d < hi - 1e-12:
        v += d
        up.append(round(v, 10))
    if abs(up[-1] - hi) > 1e-12:
        up.append(hi)

    down = []
    v = hi
    while v - d > last + 1e-12:
        v -= d
        down.append(round(v, 10))
    if not down or abs(down[-1] - last) > 1e-12:
        down.append(last)

    levels = up + down
    return [float(np.clip(x, 0.0, 1.0)) for x in levels]


def _stair_value(
    elapsed_s: float,
    *,
    levels: list[float] | tuple[float, ...],
    hold_s: float,
    transition_s: float,
) -> float:
    """
    Piecewise profile:
      level0 hold -> transition -> level1 hold -> ... -> levelN hold, then keep levelN.
    """
    t = float(max(0.0, elapsed_s))
    vals = [float(v) for v in levels]
    hold = float(max(0.0, hold_s))

    if len(vals) == 0:
        return 0.0
    if len(vals) == 1:
        return float(np.clip(vals[0], 0.0, 1.0))

    # First level hold
    if t < hold:
        return float(np.clip(vals[0], 0.0, 1.0))
    t -= hold

    tr = float(max(0.0, transition_s))
    for i in range(1, len(vals)):
        v_prev = vals[i - 1]
        v_curr = vals[i]
        if tr > 0.0:
            if t < tr:
                a = t / tr
                return float(np.clip(v_prev + (v_curr - v_prev) * a, 0.0, 1.0))
            t -= tr
        if t < hold:
            return float(np.clip(v_curr, 0.0, 1.0))
        t -= hold

    return float(np.clip(vals[-1], 0.0, 1.0))


def _build_stair_ctrl(
    elapsed_s: float,
    *,
    n_ctrl: int,
    levels: list[float] | tuple[float, ...],
    hold_s: float,
    transition_s: float,
    phase_offsets_s: list[float] | tuple[float, ...],
) -> np.ndarray:
    if len(phase_offsets_s) != n_ctrl:
        raise ValueError(
            f"phase_offsets_s length must be {n_ctrl}, got {len(phase_offsets_s)}"
        )
    out = np.zeros(n_ctrl, dtype=np.float64)
    for i in range(n_ctrl):
        out[i] = _stair_value(
            elapsed_s + float(phase_offsets_s[i]),
            levels=levels,
            hold_s=hold_s,
            transition_s=transition_s,
        )
    return np.clip(out, 0.0, 1.0)


def read_flowrate_from_obs_json(obs_json_path: str) -> tuple[float, float, float, float, float, float]:
    """obs_act.json에서 flowrate1~6을 읽는다. 없으면 NaN 반환."""
    try:
        with open(obs_json_path, "r", encoding="utf-8") as f:
            obs = json.load(f)
        fr1 = float(obs.get("flowrate1", float("nan")))
        fr2 = float(obs.get("flowrate2", float("nan")))
        fr3 = float(obs.get("flowrate3", float("nan")))
        fr4 = float(obs.get("flowrate4", float("nan")))
        fr5 = float(obs.get("flowrate5", float("nan")))
        fr6 = float(obs.get("flowrate6", float("nan")))
        return fr1, fr2, fr3, fr4, fr5, fr6
    except Exception:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan


def main():
    # -----------------------------
    # 설정 (필요 시 여기만 수정)
    # -----------------------------
    freq = 200.0                 # control freq [Hz]
    duration = 500.0            # experiment duration [sec]
    tag = ""                    # filename tag suffix

    # main chamber control mode: "zero" | "const" | "random" (0~1 range, same distribution as actuator)
    main_mode = MAIN_MODE
    main_ctrls = [0.0, 0.0]     # used when main_mode == "const" (0~1)

    # actuator control mode: "zero" | "const" | "random"  (0~1 range)
    act_mode = ACT_MODE
    act_ctrls = [0.0, 0.0, 0.0, 0.0]   # used when act_mode == "const"

    # random mode settings (shared distribution)
    rand_hold_min = int(RAND_HOLD_MIN)   # hold time range [sec] (inclusive)
    rand_hold_max = int(RAND_HOLD_MAX)
    rand_min = float(RAND_MIN)
    rand_max = float(RAND_MAX)

    if main_mode not in ("zero", "const", "random", "stair"):
        raise ValueError(f"MAIN_MODE must be zero|const|random|stair, got: {main_mode}")
    if act_mode not in ("zero", "const", "random", "stair"):
        raise ValueError(f"ACT_MODE must be zero|const|random|stair, got: {act_mode}")
    if rand_hold_min < 1 or rand_hold_max < rand_hold_min:
        raise ValueError("RAND_HOLD_MIN/MAX must satisfy 1 <= min <= max")
    if not (0.0 <= rand_min <= 1.0 and 0.0 <= rand_max <= 1.0):
        raise ValueError("RAND_MIN/RAND_MAX must be in [0,1]")
    if rand_min > rand_max:
        raise ValueError("RAND_MIN must be <= RAND_MAX")
    if STAIR_HOLD_SEC < 0.0:
        raise ValueError("STAIR_HOLD_SEC must be >= 0")
    if STAIR_TRANSITION_SEC < 0.0:
        raise ValueError("STAIR_TRANSITION_SEC must be >= 0")
    if not (0.0 <= STAIR_START <= 1.0 and 0.0 <= STAIR_PEAK <= 1.0 and 0.0 <= STAIR_END <= 1.0):
        raise ValueError("STAIR_START/PEAK/END must be in [0,1]")
    if STAIR_DELTA <= 0.0:
        raise ValueError("STAIR_DELTA must be > 0")
    if len(MAIN_STAIR_PHASE_SEC) != 2:
        raise ValueError("MAIN_STAIR_PHASE_SEC must have 2 values")
    if len(ACT_STAIR_PHASE_SEC) != 4:
        raise ValueError("ACT_STAIR_PHASE_SEC must have 4 values")
    stair_levels = _make_stair_levels(STAIR_START, STAIR_PEAK, STAIR_END, STAIR_DELTA)

    tag = f"_{tag}" if tag else ""
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    save_file_name = f"{formatted_time}_Flowrate_RND6_{main_mode}_{act_mode}{tag}"

    use_scale = bool(USE_SCALE)
    env = PneuRealAct(freq=freq, scale=use_scale)
    print(
        f"[INFO] mode: main={main_mode}, act={act_mode}, "
        f"rand_hold=[{rand_hold_min},{rand_hold_max}]s, rand_range=[{rand_min},{rand_max}], "
        f"stair(start={STAIR_START}, peak={STAIR_PEAK}, end={STAIR_END}, "
        f"delta={STAIR_DELTA}, levels={stair_levels}, "
        f"hold={STAIR_HOLD_SEC}s, trans={STAIR_TRANSITION_SEC}s, "
        f"main_phase={MAIN_STAIR_PHASE_SEC}, act_phase={ACT_STAIR_PHASE_SEC}), "
        f"scale={use_scale}"
    )

    dummy_goal = np.array([ATM, 0.0], dtype=np.float64)
    if main_mode == "const":
        main_ctrls = np.array(main_ctrls, dtype=np.float64)
    else:
        main_ctrls = np.zeros(2, dtype=np.float64)
    main_ctrls = np.clip(main_ctrls, 0.0, 1.0)

    if act_mode == "const":
        act_ctrls = np.array(act_ctrls, dtype=np.float64)
    else:
        act_ctrls = np.zeros(4, dtype=np.float64)
    act_ctrls = np.clip(act_ctrls, 0.0, 1.0)
    next_main_change = 0.0
    next_act_change = 0.0

    obs_json_path = os.path.join(get_pkg_path("pneu_env"), "tcpip/obs_act.json")

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
        curr_time = 0.0
        prev_time = None
        started = False
        start_obs_time = None
        wait_print_next_wall = 0.0

        while True:
            elapsed = 0.0 if (not started or start_obs_time is None) else max(0.0, curr_time - start_obs_time)
            if started:
                if main_mode == "random":
                    if curr_time >= next_main_change:
                        main_ctrls = np.random.uniform(
                            low=rand_min,
                            high=rand_max,
                            size=2,
                        ).astype(np.float64)
                        main_ctrls = np.clip(main_ctrls, 0.0, 1.0)
                        hold = float(np.random.randint(rand_hold_min, rand_hold_max + 1))
                        next_main_change = curr_time + hold
                elif main_mode == "zero":
                    main_ctrls = np.zeros(2, dtype=np.float64)
                elif main_mode == "stair":
                    main_ctrls = _build_stair_ctrl(
                        elapsed,
                        n_ctrl=2,
                        levels=stair_levels,
                        hold_s=float(STAIR_HOLD_SEC),
                        transition_s=float(STAIR_TRANSITION_SEC),
                        phase_offsets_s=MAIN_STAIR_PHASE_SEC,
                    )
            else:
                # time reset/fresh start 확인 전에는 랜덤 제어를 막고 정지 입력만 송신
                main_ctrls = np.zeros(2, dtype=np.float64)

            # PneuRealAct expects ctrl in [-1,1]; scale back so final valve cmd is main_ctrls (0~1)
            curr_ctrl = 2.0 * np.asarray(main_ctrls, dtype=np.float64) - 1.0

            if started:
                if act_mode == "random":
                    if curr_time >= next_act_change:
                        act_ctrls = np.random.uniform(
                            low=rand_min,
                            high=rand_max,
                            size=4,
                        ).astype(np.float64)
                        act_ctrls = np.clip(act_ctrls, 0.0, 1.0)
                        hold = float(np.random.randint(rand_hold_min, rand_hold_max + 1))
                        next_act_change = curr_time + hold
                elif act_mode == "zero":
                    act_ctrls = np.zeros(4, dtype=np.float64)
                elif act_mode == "stair":
                    act_ctrls = _build_stair_ctrl(
                        elapsed,
                        n_ctrl=4,
                        levels=stair_levels,
                        hold_s=float(STAIR_HOLD_SEC),
                        transition_s=float(STAIR_TRANSITION_SEC),
                        phase_offsets_s=ACT_STAIR_PHASE_SEC,
                    )
            else:
                act_ctrls = np.zeros(4, dtype=np.float64)
            obs, info = env.observe(ctrl=curr_ctrl, goal=dummy_goal, act_ctrls=act_ctrls)

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
                    start_obs_time = obs_time
                    for k in data.keys():
                        data[k].clear()
                    next_main_change = obs_time
                    next_act_change = obs_time
                    print(f"[INFO] fresh start detected (time={obs_time:.3f}), start logging.")
            else:
                if not started:
                    if obs_time < prev_time - 1e-3:
                        started = True
                        start_obs_time = obs_time
                        for k in data.keys():
                            data[k].clear()
                        # time reset -> restart random schedule
                        next_main_change = obs_time
                        next_act_change = obs_time
                        print(f"[INFO] time reset detected ({prev_time:.3f} -> {obs_time:.3f}), start logging.")
                    elif (
                        0.0 <= prev_time <= 1.0
                        and obs_time > prev_time + 1e-6
                        and obs_time <= 2.0
                    ):
                        started = True
                        start_obs_time = prev_time
                        for k in data.keys():
                            data[k].clear()
                        next_main_change = obs_time
                        next_act_change = obs_time
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

            curr_time = float(obs[0])

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping experiment")

    finally:
        # 안전하게 밸브를 열어 놓고 종료
        try:
            env.observe(
                ctrl=np.array([1.0, 1.0], dtype=np.float64),
                goal=dummy_goal,
                act_ctrls=np.zeros(4, dtype=np.float64),
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
