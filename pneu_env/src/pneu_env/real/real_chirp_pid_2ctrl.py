#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_chirp_pid_2ctrl.py

2-channel valve identification runner.

- Uses only ctrl1/ctrl2.
- One channel applies a chirp or stepped-sine command.
- The other channel applies PID to hold a pressure operating point.
- Runtime ref/gain/chirp settings are reloaded from a JSON file every loop.

Edit the config block below for defaults. While running, edit:
  <tcpip_dir>/chirp_pid_2ctrl_live.json
to change PID ref/gains or profile parameters without restarting.
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
DURATION = 800
TAG = "fixed_sine_pid_2ctrl"

# 1 means ctrl1/pos_ctrl, 2 means ctrl2/neg_ctrl.
CHIRP_CHANNEL = 1
PID_CHANNEL = 2

# Channels 3~6 are held fixed.
FIXED_TAIL_CTRLS = [0.0, 0.0, 0.0, 0.0]

# Safe command written before start and on exit.
SAFE_CTRLS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

LIVE_CONFIG_NAME = "fixed_sine_pid_2ctrl_live.json"

DEFAULT_LIVE_CONFIG = dict(
    enabled=True,
    # profile="chirp" or "stepped_sine".
    profile="stepped_sine",
    # Chirp command: offset + amp * sin(2*pi*(f0*t + 0.5*k*t^2) + phase)
    chirp=dict(
        # offset=0.925,
        # amp=0.025,
        # f0=0.05,
        # f1=2.0,
        # duration=180.0,
        # repeat=True,
        # phase=0.0,
        # min=0.85,
        # max=1.0,
        offset=0.925,
        amp=0.01,
        f0=0.05,
        f1=2.0,
        duration=240.0,
        repeat=False,
        phase=0.0,
        min=0.85,
        max=1.0,
        # offset=0.925,
        # amp=0.01,
        # f0=0.05,
        # f1=5.0,
        # duration=300
    ),
    # Stepped sine command:
    #   offset + amp*sin(2*pi*freq*t_local + phase)
    # If hold_sec is set, each frequency is held for exactly hold_sec seconds.
    # Otherwise each frequency is held for max(cycles_per_freq / freq, min_hold_sec).
    # The first discard_cycles cycles can be ignored later during analysis.
    stepped_sine=dict(
        offset=0.95,
        amp=0.05,
        freqs=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        hold_sec=100.0,
        cycles_per_freq=5.0,
        discard_cycles=2.0,
        startup_discard_sec=20.0,
        min_hold_sec=5.0,
        repeat=False,
        phase=0.0,
        min=0.85,
        max=1.0,
    ),
    # PID command: base + sign*(kp*err + ki*int(err) + kd*derr)
    # err = ref - measured_pressure
    # sign=-1 opens the valve when measured_pressure is above ref.
    pid=dict(
        measure_key="pos_press",
        ref=260.0,
        base=0.925,
        kp=0.003,
        ki=0.0,
        kd=0.0,
        sign=-1.0,
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


def _stepped_sine_schedule(cfg):
    freqs = [float(freq) for freq in cfg.get("freqs", [])]
    if not freqs:
        raise ValueError("stepped_sine freqs must not be empty")
    if any(freq <= 0.0 for freq in freqs):
        raise ValueError("stepped_sine freqs must be positive")

    hold_sec_cfg = cfg.get("hold_sec", None)
    hold_sec = None if hold_sec_cfg is None else float(hold_sec_cfg)
    if hold_sec is not None and hold_sec <= 0.0:
        raise ValueError("stepped_sine hold_sec must be positive when set")

    cycles_per_freq = float(cfg.get("cycles_per_freq", 5.0))
    min_hold_sec = float(cfg.get("min_hold_sec", 0.0))

    schedule = []
    start = 0.0
    for idx, freq in enumerate(freqs):
        segment_hold_sec = hold_sec
        if segment_hold_sec is None:
            segment_hold_sec = max(cycles_per_freq / freq, min_hold_sec)
        end = start + segment_hold_sec
        schedule.append(
            dict(
                idx=idx,
                freq=freq,
                start=start,
                end=end,
                hold_sec=segment_hold_sec,
            )
        )
        start = end
    return schedule, start


def _stepped_sine_value(local_time, cfg):
    schedule, total_duration = _stepped_sine_schedule(cfg)
    if bool(cfg.get("repeat", False)):
        t = local_time % max(total_duration, 1e-9)
    else:
        t = min(local_time, max(total_duration - 1e-9, 0.0))

    segment = schedule[-1]
    for item in schedule:
        if item["start"] <= t < item["end"]:
            segment = item
            break

    freq = float(segment["freq"])
    t_local = t - float(segment["start"])
    phase = 2.0 * np.pi * freq * t_local + float(cfg.get("phase", 0.0))
    ctrl = float(cfg["offset"]) + float(cfg["amp"]) * np.sin(phase)
    ctrl = float(np.clip(ctrl, float(cfg["min"]), float(cfg["max"])))

    discard_cycles = float(cfg.get("discard_cycles", 2.0))
    discard_sec = discard_cycles / freq
    startup_discard_sec = float(cfg.get("startup_discard_sec", 0.0))
    usable = t_local >= discard_sec and t >= startup_discard_sec

    return ctrl, freq, dict(
        idx=float(segment["idx"]),
        local_time=float(t_local),
        hold_sec=float(segment["hold_sec"]),
        discard_sec=float(discard_sec),
        startup_discard_sec=float(startup_discard_sec),
        usable=float(usable),
        total_duration=float(total_duration),
    )


def _profile_value(local_time, cfg):
    profile = str(cfg.get("profile", "chirp")).lower()
    if profile == "chirp":
        ctrl, freq = _chirp_value(local_time, cfg["chirp"])
        info = dict(
            profile_mode=0.0,
            profile_idx=float("nan"),
            profile_local_time=float("nan"),
            profile_hold_sec=float("nan"),
            profile_discard_sec=float("nan"),
            profile_startup_discard_sec=float("nan"),
            profile_usable=1.0,
            profile_total_duration=float("nan"),
        )
        return ctrl, freq, info
    if profile == "stepped_sine":
        ctrl, freq, sine_info = _stepped_sine_value(local_time, cfg["stepped_sine"])
        info = dict(
            profile_mode=1.0,
            profile_idx=sine_info["idx"],
            profile_local_time=sine_info["local_time"],
            profile_hold_sec=sine_info["hold_sec"],
            profile_discard_sec=sine_info["discard_sec"],
            profile_startup_discard_sec=sine_info["startup_discard_sec"],
            profile_usable=sine_info["usable"],
            profile_total_duration=sine_info["total_duration"],
        )
        return ctrl, freq, info
    raise ValueError(f"unknown profile: {profile}")


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
        rt_time=deque(),
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
        profile_mode=deque(),
        profile_idx=deque(),
        profile_local_time=deque(),
        profile_hold_sec=deque(),
        profile_discard_sec=deque(),
        profile_startup_discard_sec=deque(),
        profile_usable=deque(),
        profile_total_duration=deque(),
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
    run_start_perf = next_tick
    last_elapsed = 0.0
    warned_stale_rt_time = False

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
                    f"profile={live_cfg.get('profile', 'chirp')}, "
                    f"pid_ref={live_cfg['pid']['ref']}, "
                    f"kp={live_cfg['pid']['kp']}, ki={live_cfg['pid']['ki']}, kd={live_cfg['pid']['kd']}, "
                    f"chirp_amp={live_cfg['chirp']['amp']}, f0={live_cfg['chirp']['f0']}, f1={live_cfg['chirp']['f1']}, "
                    f"stepped_freqs={live_cfg['stepped_sine']['freqs']}"
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
            elapsed = max(0.0, now_tick - run_start_perf)
            if (
                not warned_stale_rt_time
                and elapsed > 1.0
                and obs_time <= prev_obs_time + 1e-9
            ):
                print(
                    "[WARN] RT obs time is not advancing; "
                    "using local monotonic time for chirp/logging."
                )
                warned_stale_rt_time = True

            if not bool(live_cfg.get("enabled", True)):
                ctrl = safe_ctrls.copy()
                chirp_ctrl = float("nan")
                chirp_freq = float("nan")
                profile_info = dict(
                    profile_mode=float("nan"),
                    profile_idx=float("nan"),
                    profile_local_time=float("nan"),
                    profile_hold_sec=float("nan"),
                    profile_discard_sec=float("nan"),
                    profile_startup_discard_sec=float("nan"),
                    profile_usable=float("nan"),
                    profile_total_duration=float("nan"),
                )
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
                chirp_ctrl, chirp_freq, profile_info = _profile_value(elapsed, live_cfg)
                measure_key = str(live_cfg["pid"]["measure_key"])
                if measure_key not in obs_state:
                    raise KeyError(f"PID measure_key not found in obs_state: {measure_key}")
                dt_obs = max(elapsed - last_elapsed, period)
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

            data["time"].append(elapsed)
            data["rt_time"].append(obs_time)
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
            data["profile_mode"].append(float(profile_info["profile_mode"]))
            data["profile_idx"].append(float(profile_info["profile_idx"]))
            data["profile_local_time"].append(float(profile_info["profile_local_time"]))
            data["profile_hold_sec"].append(float(profile_info["profile_hold_sec"]))
            data["profile_discard_sec"].append(float(profile_info["profile_discard_sec"]))
            data["profile_startup_discard_sec"].append(float(profile_info["profile_startup_discard_sec"]))
            data["profile_usable"].append(float(profile_info["profile_usable"]))
            data["profile_total_duration"].append(float(profile_info["profile_total_duration"]))
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
                    f"profile=({chirp_ctrl:.3f}, {chirp_freq:.3f}Hz) "
                    f"pid=(ref={pid_info['ref']:.1f}, meas={pid_info['measured']:.1f}, err={pid_info['err']:.2f})"
                )

            last_elapsed = elapsed
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
