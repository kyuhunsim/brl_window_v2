#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_chirp_pid_2ctrl.py

2-channel valve identification runner.

- Uses only ctrl1/ctrl2.
- One channel applies a chirp command.
- The other channel applies PID to hold a pressure operating point.
- Runtime ref/gain/chirp settings are reloaded from a JSON file every loop.

Edit the config block below for defaults. While running, edit:
  <tcpip_dir>/chirp_pid_2ctrl_live.json
to change PID ref/gains or chirp parameters without restarting.
"""

import json
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from pneu_env.real.real_act_flowrate_6ctrl import (
    ATM,
    _build_ctrl_payload,
    _initial_obs_state,
    _read_obs_state,
    _resolve_tcpip_dir,
    _write_json_atomic,
)
from pneu_utils.utils import get_pkg_path


# ==============================
# Manual runtime config
# ==============================
FREQ = 50.0
DURATION = 300.0
TAG = "chirp_pid_2ctrl"

# 1 means ctrl1/pos_ctrl, 2 means ctrl2/neg_ctrl.
CHIRP_CHANNEL = 1
PID_CHANNEL = 2

# Channels 3~6 are held fixed.
FIXED_TAIL_CTRLS = [0.0, 0.0, 0.0, 0.0]

# Safe command written before start and on exit.
SAFE_CTRLS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

LIVE_CONFIG_NAME = "chirp_pid_2ctrl_live.json"

DEFAULT_LIVE_CONFIG = dict(
    enabled=True,
    # Chirp command: offset + amp * sin(2*pi*(f0*t + 0.5*k*t^2) + phase)
    chirp=dict(
        offset=0.925,
        amp=0.025,
        f0=0.05,
        f1=2.0,
        duration=180.0,
        repeat=True,
        phase=0.0,
        min=0.85,
        max=1.0,
    ),
    # PID command: base + sign*(kp*err + ki*int(err) + kd*derr)
    # err = ref - measured_pressure
    pid=dict(
        measure_key="pos_press",
        ref=160.0,
        base=0.925,
        kp=0.003,
        ki=0.0,
        kd=0.0,
        sign=1.0,
        min=0.85,
        max=1.0,
        integral_limit=500.0,
        reset_integral=False,
    ),
)


def _deep_update(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _read_live_config(path, prev_cfg, prev_mtime):
    if not os.path.isfile(path):
        _write_json_atomic(path, DEFAULT_LIVE_CONFIG)
        return dict(DEFAULT_LIVE_CONFIG), os.path.getmtime(path), True

    mtime = os.path.getmtime(path)
    if prev_mtime is not None and mtime <= prev_mtime:
        return prev_cfg, prev_mtime, False

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = _deep_update(DEFAULT_LIVE_CONFIG, user_cfg)
    return cfg, mtime, True


def _chirp_value(local_time, cfg):
    duration = max(float(cfg["duration"]), 1e-9)
    if bool(cfg.get("repeat", True)):
        t = local_time % duration
    else:
        t = min(local_time, duration)

    f0 = float(cfg["f0"])
    f1 = float(cfg["f1"])
    if f0 <= 0.0 or f1 <= 0.0:
        raise ValueError("chirp f0/f1 must be positive")

    k = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + float(cfg.get("phase", 0.0))
    ctrl = float(cfg["offset"]) + float(cfg["amp"]) * np.sin(phase)
    inst_freq = f0 + k * t
    return float(np.clip(ctrl, float(cfg["min"]), float(cfg["max"]))), float(inst_freq)


class LivePID:
    def __init__(self):
        self.integral = 0.0
        self.prev_err = None

    def reset(self):
        self.integral = 0.0
        self.prev_err = None

    def compute(self, measured, cfg, dt):
        ref = float(cfg["ref"])
        err = ref - float(measured)
        dt = max(float(dt), 1e-9)

        self.integral += err * dt
        integral_limit = abs(float(cfg.get("integral_limit", 500.0)))
        self.integral = float(np.clip(self.integral, -integral_limit, integral_limit))

        if self.prev_err is None:
            derr = 0.0
        else:
            derr = (err - self.prev_err) / dt
        self.prev_err = err

        p_term = float(cfg["kp"]) * err
        i_term = float(cfg["ki"]) * self.integral
        d_term = float(cfg["kd"]) * derr
        raw = float(cfg["base"]) + float(cfg["sign"]) * (p_term + i_term + d_term)
        ctrl = float(np.clip(raw, float(cfg["min"]), float(cfg["max"])))

        return ctrl, dict(
            ref=ref,
            measured=float(measured),
            err=err,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            raw=raw,
            integral=self.integral,
        )


def _compose_ctrls(ctrl1, ctrl2):
    tail = np.asarray(FIXED_TAIL_CTRLS, dtype=np.float64)
    if tail.shape != (4,):
        raise ValueError(f"FIXED_TAIL_CTRLS must have 4 values, got {tail.shape}")
    ctrl = np.concatenate([np.array([ctrl1, ctrl2], dtype=np.float64), tail])
    return np.clip(ctrl, 0.0, 1.0)


def _active_ctrls(chirp_ctrl, pid_ctrl):
    if CHIRP_CHANNEL == 1 and PID_CHANNEL == 2:
        return float(chirp_ctrl), float(pid_ctrl)
    if CHIRP_CHANNEL == 2 and PID_CHANNEL == 1:
        return float(pid_ctrl), float(chirp_ctrl)
    raise ValueError("CHIRP_CHANNEL/PID_CHANNEL must be 1 and 2 in either order")


def main():
    freq = float(FREQ)
    period = 1.0 / max(freq, 1e-9)
    duration = float(DURATION)

    tcpip_dir = _resolve_tcpip_dir()
    ctrl_json_path = os.path.join(tcpip_dir, "ctrl_act.json")
    obs_json_path = os.path.join(tcpip_dir, "obs_act.json")
    live_config_path = os.path.join(tcpip_dir, LIVE_CONFIG_NAME)

    now = datetime.now()
    save_file_name = f"{now.strftime('%y%m%d_%H_%M_%S')}_{TAG}"
    exp_dir = os.path.join(get_pkg_path("pneu_env"), "exp")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[INFO] tcpip dir: {tcpip_dir}")
    print(f"[INFO] live config: {live_config_path}")
    print(f"[INFO] chirp_channel=ctrl{CHIRP_CHANNEL}, pid_channel=ctrl{PID_CHANNEL}")
    print(f"[INFO] duration={duration:.1f}s, freq={freq:.1f}Hz")

    obs_state = _initial_obs_state()
    goal = np.array([ATM, 0.0], dtype=np.float64)
    pid = LivePID()
    live_cfg = dict(DEFAULT_LIVE_CONFIG)
    live_mtime = None

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
        angle=deque(),
        angle_vel=deque(),
        chirp_ctrl=deque(),
        chirp_freq=deque(),
        pid_ctrl=deque(),
        pid_ref=deque(),
        pid_measured=deque(),
        pid_err=deque(),
        pid_p=deque(),
        pid_i=deque(),
        pid_d=deque(),
        pid_raw=deque(),
        pid_integral=deque(),
    )

    script_start_time = time.time()
    next_tick = time.perf_counter()
    last_obs_time = float(obs_state["time"])
    run_start_obs_time = None

    try:
        safe_ctrls = np.asarray(SAFE_CTRLS, dtype=np.float64)
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

        while True:
            now_tick = time.perf_counter()
            if now_tick < next_tick:
                time.sleep(next_tick - now_tick)
                now_tick = time.perf_counter()
            elif now_tick - next_tick > 2.0 * period:
                next_tick = now_tick
            next_tick += period

            live_cfg, live_mtime, changed = _read_live_config(
                live_config_path,
                live_cfg,
                live_mtime,
            )
            if changed:
                print(
                    "[INFO] live config loaded: "
                    f"pid_ref={live_cfg['pid']['ref']}, "
                    f"kp={live_cfg['pid']['kp']}, ki={live_cfg['pid']['ki']}, kd={live_cfg['pid']['kd']}, "
                    f"chirp_amp={live_cfg['chirp']['amp']}, f0={live_cfg['chirp']['f0']}, f1={live_cfg['chirp']['f1']}"
                )
                if bool(live_cfg["pid"].get("reset_integral", False)):
                    pid.reset()
                    print("[INFO] PID integral reset")

            prev_obs_time = float(obs_state["time"])
            obs_state = _read_obs_state(
                obs_json_path=obs_json_path,
                prev_state=obs_state,
                sen_period=period,
                max_wait_s=0.6 * period,
            )
            obs_time = float(obs_state["time"])
            if run_start_obs_time is None:
                run_start_obs_time = obs_time
            elapsed = max(0.0, obs_time - run_start_obs_time)

            if not bool(live_cfg.get("enabled", True)):
                ctrl = safe_ctrls.copy()
                chirp_ctrl = float("nan")
                chirp_freq = float("nan")
                pid_ctrl = float("nan")
                pid_info = dict(
                    ref=float("nan"),
                    measured=float("nan"),
                    err=float("nan"),
                    p_term=float("nan"),
                    i_term=float("nan"),
                    d_term=float("nan"),
                    raw=float("nan"),
                    integral=pid.integral,
                )
            else:
                chirp_ctrl, chirp_freq = _chirp_value(elapsed, live_cfg["chirp"])
                measure_key = str(live_cfg["pid"]["measure_key"])
                if measure_key not in obs_state:
                    raise KeyError(f"PID measure_key not found in obs_state: {measure_key}")
                dt_obs = max(obs_time - last_obs_time, period)
                pid_ctrl, pid_info = pid.compute(obs_state[measure_key], live_cfg["pid"], dt_obs)
                ctrl1, ctrl2 = _active_ctrls(chirp_ctrl, pid_ctrl)
                ctrl = _compose_ctrls(ctrl1, ctrl2)

            _write_json_atomic(
                ctrl_json_path,
                _build_ctrl_payload(
                    obs_state=obs_state,
                    goal=goal,
                    ctrl_unit=ctrl[:2],
                    act_unit=ctrl[2:],
                    start_time=script_start_time,
                ),
            )

            data["time"].append(obs_time)
            data["press_pos"].append(float(obs_state["pos_press"]))
            data["press_neg"].append(float(obs_state["neg_press"]))
            data["act_pos_press"].append(float(obs_state["act_pos_press"]))
            data["act_neg_press"].append(float(obs_state["act_neg_press"]))
            for i in range(6):
                data[f"ctrl{i + 1}"].append(float(ctrl[i]))
                data[f"flow{i + 1}"].append(float(obs_state[f"flowrate{i + 1}"]))
            data["angle"].append(float(obs_state["angle"]))
            data["angle_vel"].append(float(obs_state["angular_vel"]))
            data["chirp_ctrl"].append(float(chirp_ctrl))
            data["chirp_freq"].append(float(chirp_freq))
            data["pid_ctrl"].append(float(pid_ctrl))
            data["pid_ref"].append(float(pid_info["ref"]))
            data["pid_measured"].append(float(pid_info["measured"]))
            data["pid_err"].append(float(pid_info["err"]))
            data["pid_p"].append(float(pid_info["p_term"]))
            data["pid_i"].append(float(pid_info["i_term"]))
            data["pid_d"].append(float(pid_info["d_term"]))
            data["pid_raw"].append(float(pid_info["raw"]))
            data["pid_integral"].append(float(pid_info["integral"]))

            if len(data["time"]) % max(int(freq), 1) == 0:
                print(
                    f"[INFO] t={elapsed:.2f}s "
                    f"ctrl=({ctrl[0]:.3f},{ctrl[1]:.3f}) "
                    f"P=({obs_state['pos_press']:.1f},{obs_state['neg_press']:.1f}) "
                    f"chirp=({chirp_ctrl:.3f}, {chirp_freq:.3f}Hz) "
                    f"pid=(ref={pid_info['ref']:.1f}, meas={pid_info['measured']:.1f}, err={pid_info['err']:.2f})"
                )

            last_obs_time = obs_time
            if elapsed >= duration:
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping experiment")

    finally:
        try:
            safe_ctrls = np.asarray(SAFE_CTRLS, dtype=np.float64)
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

        df = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in data.items()})
        out_path = os.path.join(exp_dir, f"{save_file_name}.csv")
        df.to_csv(out_path, index=False)

        cfg_path = os.path.join(exp_dir, f"{save_file_name}_cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                dict(
                    freq=FREQ,
                    duration=DURATION,
                    chirp_channel=CHIRP_CHANNEL,
                    pid_channel=PID_CHANNEL,
                    fixed_tail_ctrls=FIXED_TAIL_CTRLS,
                    default_live_config=DEFAULT_LIVE_CONFIG,
                    live_config_path=live_config_path,
                ),
                f,
                indent=2,
            )

        print(f"[INFO] Saved experiment CSV: {out_path}")
        print(f"[INFO] Saved config JSON: {cfg_path}")


if __name__ == "__main__":
    main()
