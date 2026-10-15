#!/usr/bin/env python3
"""Build a pressure-space lib3 reference and certify it against the plant.

This script deliberately separates two objects that were conflated by the
open-loop oracle generator:

* the training reference is a smooth random pressure target; and
* simulator/PID commands are audit data used only to accept or reject it.

The pressure target is made from independent POS/NEG minimum-jerk waypoint
schedules.  Transition times are selected from signed local pressure-rate
capacity measured with a full six-valve sweep of the lib3 simulator.  A
reserve PID replay then checks trend direction, pressure bounds, chamber
reserve, and tracking quality.  The training NPZ contains only ``time`` and
``pressure``.  It never contains a teacher/oracle action.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_utils" / "src",
    SCRIPT_DIR,
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from baseline_lib3 import BASELINE_CONFIG, chamber_allocator, pid_allocator
from pneu_env.sim3 import PneuSim


ATM = 101.325
MIN_JERK_PEAK_SLOPE = 1.875
PRESSURE_NAMES = ("ch_pos", "ch_neg", "act_pos", "act_neg")
ACTUATOR_NAMES = ("act_pos", "act_neg")

# This is the explicitly recorded, previously tested lib3 PID used only for
# feasibility screening.  The slower historical baseline PID is retained as
# a comparison result, but it is not a sound reachability oracle: on a smooth
# atmospheric startup its small error produces too little inlet opening.  No
# PID quantity is exposed to the RL policy.
CERTIFIER_PID = {
    "act_pos_in": {"kp": 0.09, "ki": 0.005, "kd": 0.001},
    "act_pos_out": {"kp": 0.052, "ki": 0.00045, "kd": 0.001},
    "act_neg_in": {"kp": 0.058, "ki": 0.00135, "kd": 0.001},
    "act_neg_out": {"kp": 0.075, "ki": 0.0006, "kd": 0.001},
    "deadband": 0.5,
    "integral_limit": 100.0,
    "max_open": 1.0,
}
CERTIFIER_ACTIVE_MIN_PHYSICAL = 0.88


def parse_float_pair(text: str) -> tuple[float, float]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 2 or values[0] >= values[1]:
        raise argparse.ArgumentTypeError("expected an ascending pair, e.g. 110,145")
    return values


def parse_int_pair(text: str) -> tuple[int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 2 or values[0] <= 0 or values[0] > values[1]:
        raise argparse.ArgumentTypeError("expected a positive ascending integer pair")
    return values


def quintic(progress: np.ndarray) -> np.ndarray:
    tau = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def make_sim(*, delay: float = 0.1) -> PneuSim:
    return PneuSim(
        freq=50.0,
        delay=delay,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=0.8,
        scale_high=1.0,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )


def operating_states(
    pos_bounds: tuple[float, float],
    neg_bounds: tuple[float, float],
    margin: float,
) -> list[tuple[float, float, float, float]]:
    """Return atmosphere plus a grid spanning the intended operating box."""
    states: list[tuple[float, float, float, float]] = [(ATM, ATM, ATM, ATM)]
    for act_pos in np.linspace(pos_bounds[0] + 2.0, pos_bounds[1] - 2.0, 3):
        for act_neg in np.linspace(neg_bounds[0] + 2.0, neg_bounds[1] - 2.0, 3):
            states.append(
                (
                    float(np.clip(act_pos + margin, ATM, 205.0)),
                    float(np.clip(act_neg - margin, 10.0, ATM)),
                    float(act_pos),
                    float(act_neg),
                )
            )
    return states


def run_capacity_sweep(
    states: list[tuple[float, float, float, float]],
    *,
    grid_points: int,
    warmup_sec: float,
    measure_sec: float,
) -> pd.DataFrame:
    """Measure local signed rates over a common [0.8, 1.0] valve domain."""
    sim = make_sim(delay=0.1)
    action_grid = np.linspace(0.0, 1.0, grid_points, dtype=np.float64)
    actions = list(itertools.product(action_grid, repeat=6))
    warmup_steps = max(int(round(warmup_sec * 50.0)), 6)
    measure_steps = max(int(round(measure_sec * 50.0)), 5)
    rows: list[dict[str, float]] = []

    for state_id, state in enumerate(states):
        for action_tuple in actions:
            action = np.asarray(action_tuple, dtype=np.float64)
            sim.set_init_press(*state)
            for _ in range(warmup_steps):
                sim.observe(action)
            start = np.asarray(sim.obs, dtype=np.float64).copy()
            for _ in range(measure_steps):
                sim.observe(action)
            end = np.asarray(sim.obs, dtype=np.float64).copy()
            rate = (end - start) / (measure_steps / 50.0)
            row: dict[str, float] = {
                "state_id": float(state_id),
                "init_ch_pos": state[0],
                "init_ch_neg": state[1],
                "init_act_pos": state[2],
                "init_act_neg": state[3],
                **{
                    f"rate_{name}_kpa_s": float(rate[index])
                    for index, name in enumerate(PRESSURE_NAMES)
                },
            }
            for index, value in enumerate(action):
                row[f"u{index + 1}_normalized"] = float(value)
                row[f"u{index + 1}_physical"] = float(0.8 + 0.2 * value)
            rows.append(row)
    return pd.DataFrame(rows)


def directional_capacity(values: np.ndarray, quantile: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0.0]
    negative = -values[values < 0.0]
    return {
        "up": float(np.quantile(positive, quantile)) if len(positive) else 0.0,
        "down": float(np.quantile(negative, quantile)) if len(negative) else 0.0,
    }


def summarize_capacity(df: pd.DataFrame, quantile: float) -> dict[str, Any]:
    per_state: dict[str, Any] = {}
    for state_id, group in df.groupby("state_id"):
        item: dict[str, Any] = {
            "initial_pressure_kpa": [
                float(group.iloc[0][f"init_{name}"]) for name in PRESSURE_NAMES
            ]
        }
        for name in PRESSURE_NAMES:
            item[name] = directional_capacity(
                group[f"rate_{name}_kpa_s"].to_numpy(), quantile
            )
        per_state[str(int(state_id))] = item

    # State 0 is the atmospheric startup transient.  Steady training limits
    # are derived from the other operating points; startup is reported and
    # handled separately.
    steady_ids = [key for key in per_state if key != "0"]
    robust: dict[str, Any] = {}
    for name in PRESSURE_NAMES:
        robust[name] = {}
        for direction in ("up", "down"):
            capacities = np.asarray(
                [per_state[key][name][direction] for key in steady_ids],
                dtype=np.float64,
            )
            robust[name][direction] = {
                "min": float(np.min(capacities)),
                "q20": float(np.quantile(capacities, 0.2)),
                "median": float(np.median(capacities)),
                "max": float(np.max(capacities)),
            }
    return {
        "capacity_quantile_within_state": float(quantile),
        "per_state": per_state,
        "robust_across_steady_states": robust,
    }


def derive_reference_rates(
    summary: dict[str, Any],
    *,
    utilization: float,
    caps: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Apply a conservative fraction to q20 state-wise valve capacity."""
    robust = summary["robust_across_steady_states"]
    rates: dict[str, dict[str, float]] = {}
    for name in ACTUATOR_NAMES:
        rates[name] = {}
        for direction in ("up", "down"):
            measured = float(robust[name][direction]["q20"])
            rates[name][direction] = float(
                min(caps[name][direction], utilization * measured)
            )
            if rates[name][direction] <= 0.25:
                raise RuntimeError(
                    f"no useful {name} {direction} capacity in the intended pressure box"
                )
    return rates


def allocate_bounded_sum(
    rng: random.Random,
    count: int,
    total: float,
    lower: float,
    upper: float,
) -> list[float]:
    if total < count * lower - 1e-9 or total > count * upper + 1e-9:
        raise ValueError("bounded allocation is infeasible")
    result = [lower] * count
    remaining = total - count * lower
    indices = list(range(count))
    rng.shuffle(indices)
    for position, index in enumerate(indices):
        future_capacity = (count - position - 1) * (upper - lower)
        minimum = max(0.0, remaining - future_capacity)
        maximum = min(upper - lower, remaining)
        increment = maximum if position == len(indices) - 1 else rng.uniform(minimum, maximum)
        result[index] += increment
        remaining -= increment
    return result


@dataclass
class AxisSchedule:
    values: np.ndarray
    segments: list[dict[str, float | int]]


def sample_next_target(
    rng: random.Random,
    current: float,
    bounds: tuple[float, float],
    min_delta: float,
    max_delta: float,
    previous_direction: int,
) -> tuple[float, int]:
    low, high = bounds
    if rng.random() < 0.20:
        endpoint = rng.choice((low, high))
        endpoint_delta = abs(endpoint - current)
        if min_delta <= endpoint_delta <= max_delta:
            return float(endpoint), int(1 if endpoint > current else -1)
    preferred = -previous_direction if previous_direction else rng.choice((-1, 1))
    for attempt in range(80):
        direction = preferred if attempt < 44 else rng.choice((-1, 1))
        delta = rng.uniform(min_delta, max_delta)
        target = current + direction * delta
        if low <= target <= high:
            return float(target), int(direction)
        # Occasionally use a far boundary to cover the complete envelope,
        # while still respecting the configured maximum excursion.
        endpoint = low if direction < 0 else high
        if min_delta <= abs(endpoint - current) <= max_delta:
            return float(endpoint), int(direction)
    available = [(low, -1), (high, 1)]
    available.sort(key=lambda item: abs(item[0] - current), reverse=True)
    for target, direction in available:
        if abs(target - current) >= min_delta:
            return float(target), int(direction)
    raise RuntimeError("axis bounds cannot provide the requested minimum excursion")


def make_axis_block(
    rng: random.Random,
    *,
    block: int,
    block_start: float,
    block_duration: float,
    current: float,
    previous_direction: int,
    bounds: tuple[float, float],
    rates: dict[str, float],
    startup_target: tuple[float, float],
    startup_rate: float,
    event_count: tuple[int, int],
    min_delta: float,
    max_delta: float,
    min_transition: float,
    max_frequency_hz: float,
    hold_bounds: tuple[float, float],
) -> tuple[list[dict[str, float | int]], float, int]:
    minimum_bandwidth_duration = 1.0 / (2.0 * max_frequency_hz)
    for _ in range(400):
        count = rng.randint(event_count[0], event_count[1])
        if block == 0:
            count = max(event_count[0], min(count, event_count[1]))
        targets: list[float] = []
        durations: list[float] = []
        starts: list[float] = []
        directions: list[int] = []
        local_current = float(current)
        local_direction = int(previous_direction)

        for event in range(count):
            if block == 0 and event == 0 and not (bounds[0] <= local_current <= bounds[1]):
                target = rng.uniform(*startup_target)
                direction = 1 if target >= local_current else -1
                rate = startup_rate
            else:
                target, direction = sample_next_target(
                    rng,
                    local_current,
                    bounds,
                    min_delta,
                    max_delta,
                    local_direction,
                )
                rate = rates["up" if direction > 0 else "down"]
            delta = abs(target - local_current)
            duration = max(
                min_transition,
                minimum_bandwidth_duration,
                MIN_JERK_PEAK_SLOPE * delta / rate,
            )
            # Slower-than-capacity transitions add timing variation without
            # changing the admissible pressure-rate envelope.
            duration *= rng.uniform(1.0, 1.12)
            starts.append(local_current)
            targets.append(target)
            durations.append(duration)
            directions.append(direction)
            local_current = target
            local_direction = direction

        minimum_total = sum(durations) + count * hold_bounds[0]
        if minimum_total > block_duration:
            continue
        hold_total = min(
            block_duration - sum(durations),
            count * hold_bounds[1],
        )
        if hold_total < count * hold_bounds[0] - 1e-9:
            continue
        holds = allocate_bounded_sum(
            rng, count, hold_total, hold_bounds[0], hold_bounds[1]
        )
        remaining = block_duration - sum(durations) - sum(holds)
        if remaining > 1e-9:
            # Stretch all transitions proportionally.  This only lowers rate.
            scale = (sum(durations) + remaining) / sum(durations)
            durations = [value * scale for value in durations]

        cursor = block_start
        segments: list[dict[str, float | int]] = []
        for event, (start, target, duration, hold, direction) in enumerate(
            zip(starts, targets, durations, holds, directions)
        ):
            segments.append(
                {
                    "block": int(block + 1),
                    "event": int(event + 1),
                    "start_sec": float(cursor),
                    "transition_sec": float(duration),
                    "hold_sec": float(hold),
                    "pressure_start_kpa": float(start),
                    "pressure_target_kpa": float(target),
                    "direction": int(direction),
                    "end_sec": float(cursor + duration + hold),
                }
            )
            cursor += duration + hold
        if abs(cursor - (block_start + block_duration)) > 1e-7:
            raise AssertionError("axis block did not fill the complete block")
        return segments, float(local_current), int(local_direction)
    raise RuntimeError("could not fit a random schedule inside one block")


def render_axis(time: np.ndarray, initial: float, segments: list[dict[str, Any]]) -> np.ndarray:
    values = np.full_like(time, float(initial), dtype=np.float64)
    for segment in segments:
        start_time = float(segment["start_sec"])
        transition = float(segment["transition_sec"])
        transition_end = start_time + transition
        segment_end = float(segment["end_sec"])
        start = float(segment["pressure_start_kpa"])
        target = float(segment["pressure_target_kpa"])
        transition_mask = (time >= start_time) & (time < transition_end)
        if np.any(transition_mask):
            values[transition_mask] = start + (target - start) * quintic(
                (time[transition_mask] - start_time) / transition
            )
        hold_mask = (time >= transition_end) & (time <= segment_end + 1e-10)
        values[hold_mask] = target
    return values


def make_pressure_candidate(
    rng: random.Random,
    args: argparse.Namespace,
    rates: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    block_duration = args.block_steps / args.freq
    total_steps = args.blocks * args.block_steps
    time = np.arange(total_steps + 1, dtype=np.float64) / args.freq
    axis_specs = {
        "pos": {
            "initial": ATM,
            "bounds": args.pos_bounds,
            "rates": rates["act_pos"],
            "startup_target": args.startup_pos,
            "startup_rate": args.startup_pos_rate,
            "min_delta": args.pos_delta[0],
            "max_delta": args.pos_delta[1],
        },
        "neg": {
            "initial": ATM,
            "bounds": args.neg_bounds,
            "rates": rates["act_neg"],
            "startup_target": args.startup_neg,
            "startup_rate": args.startup_neg_rate,
            "min_delta": args.neg_delta[0],
            "max_delta": args.neg_delta[1],
        },
    }
    columns = []
    schedule: dict[str, list[dict[str, Any]]] = {}
    for axis, spec in axis_specs.items():
        current = float(spec["initial"])
        previous_direction = 0
        segments: list[dict[str, Any]] = []
        for block in range(args.blocks):
            block_segments, current, previous_direction = make_axis_block(
                rng,
                block=block,
                block_start=block * block_duration,
                block_duration=block_duration,
                current=current,
                previous_direction=previous_direction,
                bounds=spec["bounds"],
                rates=spec["rates"],
                startup_target=spec["startup_target"],
                startup_rate=spec["startup_rate"],
                event_count=args.events_per_block,
                min_delta=spec["min_delta"],
                max_delta=spec["max_delta"],
                min_transition=args.min_transition,
                max_frequency_hz=args.max_frequency_hz,
                hold_bounds=args.hold_sec,
            )
            segments.extend(block_segments)
        columns.append(render_axis(time, float(spec["initial"]), segments))
        schedule[axis] = segments
    return time, np.column_stack(columns), schedule


def derive_chamber_reference(pressure: np.ndarray, margin: float) -> np.ndarray:
    """Return a diagnostic reserve guide, not another tracking target."""
    return np.column_stack(
        [
            np.clip(pressure[:, 0] + margin, ATM, 205.0),
            np.clip(pressure[:, 1] - margin, 10.0, ATM),
        ]
    )


def nearest_capacity_groups(
    capacity: pd.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray]]:
    states: list[np.ndarray] = []
    clouds: list[np.ndarray] = []
    for _, group in capacity.groupby("state_id"):
        states.append(
            group.iloc[0][
                ["init_ch_pos", "init_ch_neg", "init_act_pos", "init_act_neg"]
            ].to_numpy(dtype=np.float64)
        )
        clouds.append(
            group[["rate_act_pos_kpa_s", "rate_act_neg_kpa_s"]].to_numpy(
                dtype=np.float64
            )
        )
    return np.vstack(states), clouds


def joint_rate_support(
    time: np.ndarray,
    pressure: np.ndarray,
    chamber_reference: np.ndarray,
    capacity_states: np.ndarray,
    capacity_clouds: list[np.ndarray],
) -> dict[str, float]:
    dt = float(np.median(np.diff(time)))
    desired = np.column_stack(
        [
            np.gradient(pressure[:, 0], dt),
            np.gradient(pressure[:, 1], dt),
        ]
    )
    full_state = np.column_stack([chamber_reference, pressure])
    scales = np.array([50.0, 35.0, 35.0, 25.0], dtype=np.float64)
    checked = 0
    supported = 0
    for index in range(0, len(time), 2):
        request = desired[index]
        active = np.abs(request) >= np.array([0.5, 0.25])
        if not np.any(active):
            continue
        distances = np.sum(((capacity_states - full_state[index]) / scales) ** 2, axis=1)
        cloud = capacity_clouds[int(np.argmin(distances))]
        mask = np.ones(len(cloud), dtype=bool)
        for channel in range(2):
            if not active[channel]:
                continue
            if request[channel] > 0.0:
                mask &= cloud[:, channel] >= request[channel]
            else:
                mask &= cloud[:, channel] <= request[channel]
        checked += 1
        supported += int(np.any(mask))
    return {
        "checked_samples": float(checked),
        "supported_samples": float(supported),
        "support_fraction": float(supported / checked) if checked else 1.0,
    }


def run_pid_certifier(
    time: np.ndarray,
    pressure: np.ndarray,
    *,
    margin: float,
) -> dict[str, np.ndarray]:
    sim = make_sim(delay=0.1)
    sim.set_init_press(ATM, ATM, ATM, ATM)
    pid_config = copy.deepcopy(CERTIFIER_PID)
    previous_error = np.zeros(4, dtype=np.float64)
    integral_error = np.zeros(4, dtype=np.float64)
    full_pressure = np.empty((len(time), 4), dtype=np.float64)
    raw_action = np.empty((len(time), 6), dtype=np.float64)
    chamber_target = derive_chamber_reference(pressure, margin)

    full_pressure[0] = np.array([ATM, ATM, ATM, ATM], dtype=np.float64)
    raw_action[0] = np.zeros(6, dtype=np.float64)
    for index in range(1, len(time)):
        goal = pressure[index]
        chamber_ctrl, _ = chamber_allocator(
            sim.obs[0:2],
            goal,
            mode="reserve",
            chamber_raw=BASELINE_CONFIG["chamber"]["raw"],
            margin=margin,
            kp=0.04,
            deadband=1.0,
            min_open=0.15,
            max_open=0.85,
            ch_pos_min=ATM,
            ch_pos_max=205.0,
            ch_neg_min=10.0,
            ch_neg_max=ATM,
        )
        action, previous_error, integral_error = pid_allocator(
            sim.obs[2:4],
            goal,
            previous_error,
            integral_error,
            dt=1.0 / 50.0,
            pid_cfg=pid_config,
            chamber_ctrl=chamber_ctrl,
        )
        # Common valve-aware minimum for the four directional actuator
        # channels.  This belongs only to the offline PID certifier.  It does
        # not project or mask the SAC action, whose physical domain remains
        # the complete [0.8, 1.0] interval for every valve.
        active_min_raw = (CERTIFIER_ACTIVE_MIN_PHYSICAL - 0.8) / 0.2
        for directional_index in range(4):
            if previous_error[directional_index] > float(pid_config["deadband"]):
                action[directional_index + 2] = max(
                    action[directional_index + 2], active_min_raw
                )
        obs, _ = sim.observe(action, goal=goal)
        full_pressure[index] = obs[1:5]
        raw_action[index] = action
    return {
        "pressure": full_pressure,
        "raw_action": raw_action,
        "physical_action": 0.8 + 0.2 * raw_action,
        "chamber_reference": chamber_target,
    }


def trend_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    dt: float,
    threshold: float,
    max_lag_sec: float = 2.0,
) -> dict[str, float]:
    window = min(21, len(actual) - (1 - len(actual) % 2))
    if window < 5:
        smooth = actual
    else:
        if window % 2 == 0:
            window -= 1
        smooth = savgol_filter(actual, window_length=window, polyorder=2)
    ref_rate = np.gradient(reference, dt)
    act_rate = np.gradient(smooth, dt)
    best_fraction = -1.0
    best_lag = 0
    max_lag = min(int(round(max_lag_sec / dt)), len(reference) // 4)
    for lag in range(max_lag + 1):
        if lag == 0:
            requested = ref_rate
            response = act_rate
        else:
            requested = ref_rate[:-lag]
            response = act_rate[lag:]
        active = np.abs(requested) >= threshold
        if not np.any(active):
            continue
        fraction = float(
            np.mean(np.sign(requested[active]) == np.sign(response[active]))
        )
        if fraction > best_fraction:
            best_fraction = fraction
            best_lag = lag
    if best_fraction < 0.0:
        return {
            "direction_fraction": 1.0,
            "signed_rate_gain_median": 0.0,
            "best_lag_sec": 0.0,
            "event_count": 0,
            "event_direction_fraction": 1.0,
            "event_gain_median": 0.0,
        }
    if best_lag == 0:
        requested = ref_rate
        response = act_rate
    else:
        requested = ref_rate[:-best_lag]
        response = act_rate[best_lag:]
    same = (np.abs(requested) >= threshold) & (
        np.sign(requested) == np.sign(response)
    )
    gain = np.abs(response[same]) / np.maximum(np.abs(requested[same]), 1e-6)
    # Event-level direction/gain prevents a tiny same-sign numerical slope
    # from being counted as successful tracking.  Each contiguous nonzero
    # minimum-jerk transition is scored once after the identified plant lag.
    event_active = np.abs(ref_rate) >= max(0.05 * threshold, 0.02)
    padded = np.r_[False, event_active, False]
    starts = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1) - 1
    event_gains: list[float] = []
    settle_steps = int(round(0.35 / dt))
    for start, end in zip(starts, ends):
        ref_start = max(int(start) - 1, 0)
        ref_end = min(int(end) + 1, len(reference) - 1)
        response_start = min(ref_start + best_lag, len(actual) - 1)
        response_end = min(ref_end + best_lag + settle_steps, len(actual) - 1)
        ref_delta = float(reference[ref_end] - reference[ref_start])
        if abs(ref_delta) < 1e-4 or response_end <= response_start:
            continue
        actual_delta = float(smooth[response_end] - smooth[response_start])
        event_gains.append(actual_delta / ref_delta)
    event_gains_array = np.asarray(event_gains, dtype=np.float64)
    positive_event_gains = event_gains_array[event_gains_array > 0.0]
    return {
        "direction_fraction": float(best_fraction),
        "signed_rate_gain_median": float(np.median(gain)) if len(gain) else 0.0,
        "best_lag_sec": float(best_lag * dt),
        "event_count": int(len(event_gains_array)),
        "event_direction_fraction": (
            float(np.mean(event_gains_array > 0.0)) if len(event_gains_array) else 1.0
        ),
        "event_gain_median": (
            float(np.median(positive_event_gains)) if len(positive_event_gains) else 0.0
        ),
    }


def count_extrema(values: np.ndarray, prominence: float) -> tuple[int, int]:
    peaks, _ = find_peaks(values, prominence=prominence, distance=25)
    valleys, _ = find_peaks(-values, prominence=prominence, distance=25)
    return int(len(peaks)), int(len(valleys))


def candidate_metrics(
    time: np.ndarray,
    pressure: np.ndarray,
    audit: dict[str, np.ndarray],
    support: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dt = 1.0 / args.freq
    actual = audit["pressure"]
    chamber = actual[:, 0:2]
    actuator = actual[:, 2:4]
    ref_rate = np.gradient(pressure, dt, axis=0)
    ch_ref_rate = np.gradient(audit["chamber_reference"], dt, axis=0)
    error = actuator - pressure
    after_start = time >= 1.0
    after_startup_transition = time >= 5.0
    pos_trend = trend_metrics(pressure[:, 0], actuator[:, 0], dt, 1.0)
    neg_trend = trend_metrics(pressure[:, 1], actuator[:, 1], dt, 0.5)
    pos_peaks, pos_valleys = count_extrema(pressure[:, 0], args.pos_delta[0] * 0.6)
    neg_peaks, neg_valleys = count_extrema(pressure[:, 1], args.neg_delta[0] * 0.6)
    reserve_pos = chamber[:, 0] - actuator[:, 0]
    reserve_neg = actuator[:, 1] - chamber[:, 1]
    raw = audit["raw_action"]
    conflicts = np.column_stack(
        [
            (raw[:, 2] > 0.1) & (raw[:, 3] > 0.1),
            (raw[:, 4] > 0.1) & (raw[:, 5] > 0.1),
        ]
    )
    metrics: dict[str, Any] = {
        "reference": {
            "actuator_pressure_min_kpa": np.min(pressure, axis=0).tolist(),
            "actuator_pressure_max_kpa": np.max(pressure, axis=0).tolist(),
            "actuator_pressure_min_after_startup_kpa": np.min(
                pressure[after_startup_transition], axis=0
            ).tolist(),
            "actuator_pressure_max_after_startup_kpa": np.max(
                pressure[after_startup_transition], axis=0
            ).tolist(),
            "actuator_rate_min_kpa_s": np.min(ref_rate, axis=0).tolist(),
            "actuator_rate_max_kpa_s": np.max(ref_rate, axis=0).tolist(),
            "chamber_reference_min_kpa": np.min(audit["chamber_reference"], axis=0).tolist(),
            "chamber_reference_max_kpa": np.max(audit["chamber_reference"], axis=0).tolist(),
            "chamber_reference_rate_min_kpa_s": np.min(ch_ref_rate, axis=0).tolist(),
            "chamber_reference_rate_max_kpa_s": np.max(ch_ref_rate, axis=0).tolist(),
            "pos_peaks": pos_peaks,
            "pos_valleys": pos_valleys,
            "neg_peaks": neg_peaks,
            "neg_valleys": neg_valleys,
            **support,
        },
        "pid_certifier": {
            "pos_rmse_kpa": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "neg_rmse_kpa": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "pos_mae_kpa": float(np.mean(np.abs(error[:, 0]))),
            "neg_mae_kpa": float(np.mean(np.abs(error[:, 1]))),
            "pos_tolerance_5kpa_fraction": float(np.mean(np.abs(error[:, 0]) <= 5.0)),
            "neg_tolerance_5kpa_fraction": float(np.mean(np.abs(error[:, 1]) <= 5.0)),
            "pos_direction_fraction": pos_trend["direction_fraction"],
            "neg_direction_fraction": neg_trend["direction_fraction"],
            "pos_signed_rate_gain_median": pos_trend["signed_rate_gain_median"],
            "neg_signed_rate_gain_median": neg_trend["signed_rate_gain_median"],
            "pos_best_response_lag_sec": pos_trend["best_lag_sec"],
            "neg_best_response_lag_sec": neg_trend["best_lag_sec"],
            "pos_event_count": pos_trend["event_count"],
            "neg_event_count": neg_trend["event_count"],
            "pos_event_direction_fraction": pos_trend["event_direction_fraction"],
            "neg_event_direction_fraction": neg_trend["event_direction_fraction"],
            "pos_event_gain_median": pos_trend["event_gain_median"],
            "neg_event_gain_median": neg_trend["event_gain_median"],
            "chamber_pos_min_kpa": float(np.min(chamber[after_start, 0])),
            "chamber_pos_max_kpa": float(np.max(chamber[after_start, 0])),
            "chamber_neg_min_kpa": float(np.min(chamber[after_start, 1])),
            "chamber_neg_max_kpa": float(np.max(chamber[after_start, 1])),
            "pos_reserve_ge_5kpa_fraction": float(np.mean(reserve_pos[after_start] >= 5.0)),
            "neg_reserve_ge_5kpa_fraction": float(np.mean(reserve_neg[after_start] >= 5.0)),
            "inlet_outlet_conflict_fraction": np.mean(conflicts, axis=0).tolist(),
            "physical_action_min": np.min(audit["physical_action"], axis=0).tolist(),
            "physical_action_max": np.max(audit["physical_action"], axis=0).tolist(),
        },
    }
    pid = metrics["pid_certifier"]
    passed = bool(
        support["support_fraction"] >= args.min_joint_support
        and pid["pos_rmse_kpa"] <= args.max_pos_rmse
        and pid["neg_rmse_kpa"] <= args.max_neg_rmse
        and pid["pos_direction_fraction"] >= args.min_pos_direction
        and pid["neg_direction_fraction"] >= args.min_neg_direction
        and pid["chamber_pos_max_kpa"] <= 205.0
        and pid["chamber_neg_min_kpa"] >= 10.0
        and pid["pos_reserve_ge_5kpa_fraction"] >= 0.90
        and pid["neg_reserve_ge_5kpa_fraction"] >= 0.90
        and pid["pos_event_direction_fraction"] >= 0.75
        and pid["neg_event_direction_fraction"] >= 0.75
        and pid["pos_event_gain_median"] >= 0.20
        and pid["neg_event_gain_median"] >= 0.20
    )
    metrics["passed"] = passed
    metrics["score"] = float(
        pid["pos_rmse_kpa"]
        + pid["neg_rmse_kpa"]
        + 8.0 * (1.0 - pid["pos_direction_fraction"])
        + 8.0 * (1.0 - pid["neg_direction_fraction"])
        + 5.0 * (1.0 - pid["pos_event_direction_fraction"])
        + 5.0 * (1.0 - pid["neg_event_direction_fraction"])
        + 20.0 * (1.0 - support["support_fraction"])
        - 0.08
        * (
            metrics["reference"]["actuator_pressure_max_after_startup_kpa"][0]
            - metrics["reference"]["actuator_pressure_min_after_startup_kpa"][0]
        )
        - 0.12
        * (
            metrics["reference"]["actuator_pressure_max_after_startup_kpa"][1]
            - metrics["reference"]["actuator_pressure_min_after_startup_kpa"][1]
        )
    )
    return metrics


def plot_reference(
    time: np.ndarray,
    pressure: np.ndarray,
    schedule: dict[str, Any],
    rates: dict[str, dict[str, float]],
    path: Path,
    args: argparse.Namespace,
) -> None:
    chamber_reference = derive_chamber_reference(pressure, args.reserve_margin)
    dt = 1.0 / args.freq
    pressure_rate = np.gradient(pressure, dt, axis=0)
    chamber_rate = np.gradient(chamber_reference, dt, axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(time, pressure[:, 0], color="red", label="Act POS ref")
    axes[0].plot(time, pressure[:, 1], color="blue", label="Act NEG ref")
    axes[0].set_ylabel("Actuator ref [kPa]")
    axes[0].legend(ncol=2)
    axes[1].plot(
        time,
        chamber_reference[:, 0],
        color="firebrick",
        label="Ch POS reserve guide (not target)",
    )
    axes[1].plot(
        time,
        chamber_reference[:, 1],
        color="navy",
        label="Ch NEG reserve guide (not target)",
    )
    axes[1].set_ylabel("Chamber reserve guide [kPa]")
    axes[1].legend(ncol=2)
    axes[2].plot(time, pressure_rate[:, 0], color="red", label="d Act POS ref/dt")
    axes[2].plot(time, pressure_rate[:, 1], color="blue", label="d Act NEG ref/dt")
    axes[2].plot(
        time,
        chamber_rate[:, 0],
        color="salmon",
        alpha=0.45,
        label="d Ch POS guide/dt",
    )
    axes[2].plot(
        time,
        chamber_rate[:, 1],
        color="cornflowerblue",
        alpha=0.45,
        label="d Ch NEG guide/dt",
    )
    axes[2].axhline(rates["act_pos"]["up"], color="red", linestyle=":", alpha=0.5)
    axes[2].axhline(-rates["act_pos"]["down"], color="red", linestyle=":", alpha=0.5)
    axes[2].axhline(rates["act_neg"]["up"], color="blue", linestyle=":", alpha=0.5)
    axes[2].axhline(-rates["act_neg"]["down"], color="blue", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Reference rate [kPa/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(ncol=4, fontsize=9)
    for axis in axes:
        for block in range(1, args.blocks):
            axis.axvline(block * args.block_steps / args.freq, color="black", linestyle="--", alpha=0.35)
        axis.grid(True, alpha=0.3)
    fig.suptitle(
        "lib3 pressure-space capacity-constrained random carry reference\n"
        "target only — no simulator command is stored in the training profile"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_bank_overview(
    time_bank: np.ndarray,
    pressure_bank: np.ndarray,
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Show temporal diversity and pressure-rate coverage of every member."""
    dt = 1.0 / args.freq
    rate_bank = np.gradient(pressure_bank, dt, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    pos_time, neg_time, pos_phase, neg_phase = axes.ravel()
    for member in range(len(pressure_bank)):
        alpha = 0.9 if member == 0 else 0.25
        width = 2.0 if member == 0 else 0.8
        pos_time.plot(
            time_bank[member], pressure_bank[member, :, 0],
            color="red", alpha=alpha, linewidth=width,
        )
        neg_time.plot(
            time_bank[member], pressure_bank[member, :, 1],
            color="blue", alpha=alpha, linewidth=width,
        )
        stride = 4
        pos_phase.scatter(
            pressure_bank[member, ::stride, 0], rate_bank[member, ::stride, 0],
            s=3, color="red", alpha=0.10,
        )
        neg_phase.scatter(
            pressure_bank[member, ::stride, 1], rate_bank[member, ::stride, 1],
            s=3, color="blue", alpha=0.10,
        )
    for axis, label in ((pos_time, "Act POS reference [kPa]"), (neg_time, "Act NEG reference [kPa]")):
        for block in range(1, args.blocks):
            axis.axvline(block * args.block_steps / args.freq, color="black", linestyle="--", alpha=0.3)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    pos_time.set_title(f"POS: {len(pressure_bank)} certified 4×512 carry members")
    neg_time.set_title(f"NEG: {len(pressure_bank)} certified 4×512 carry members")
    pos_phase.axhline(args.pos_rate_up_cap, color="black", linestyle=":", alpha=0.5)
    pos_phase.axhline(-args.pos_rate_down_cap, color="black", linestyle=":", alpha=0.5)
    neg_phase.axhline(args.neg_rate_up_cap, color="black", linestyle=":", alpha=0.5)
    neg_phase.axhline(-args.neg_rate_down_cap, color="black", linestyle=":", alpha=0.5)
    pos_phase.set(xlabel="Act POS reference [kPa]", ylabel="dP/dt [kPa/s]", title="POS pressure-rate coverage")
    neg_phase.set(xlabel="Act NEG reference [kPa]", ylabel="dP/dt [kPa/s]", title="NEG pressure-rate coverage")
    for axis in (pos_phase, neg_phase):
        axis.grid(True, alpha=0.3)
    fig.suptitle(
        "lib3 capacity-constrained pressure target bank\n"
        "all faint curves are independent certified members; member 1 is highlighted"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pid_audit(
    time: np.ndarray,
    pressure: np.ndarray,
    audit: dict[str, np.ndarray],
    path: Path,
    args: argparse.Namespace,
) -> None:
    actual = audit["pressure"]
    dt = 1.0 / args.freq
    ref_rate = np.gradient(pressure, dt, axis=0)
    act_rate = np.gradient(actual[:, 2:4], dt, axis=0)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(time, pressure[:, 0], "k", label="REF POS")
    axes[0].plot(time, actual[:, 2], "r", label="PID ACT POS")
    axes[0].plot(time, pressure[:, 1], "k--", label="REF NEG")
    axes[0].plot(time, actual[:, 3], "b", label="PID ACT NEG")
    axes[0].set_ylabel("Actuator [kPa]")
    axes[0].legend(ncol=4)
    axes[1].plot(time, ref_rate[:, 0], "r", label="dREF POS")
    axes[1].plot(time, act_rate[:, 0], "r--", alpha=0.7, label="dACT POS")
    axes[1].plot(time, ref_rate[:, 1], "b", label="dREF NEG")
    axes[1].plot(time, act_rate[:, 1], "b--", alpha=0.7, label="dACT NEG")
    axes[1].set_ylabel("Rate [kPa/s]")
    axes[1].legend(ncol=4)
    axes[2].plot(time, actual[:, 0], color="red", label="Ch POS actual")
    axes[2].plot(time, actual[:, 1], color="blue", label="Ch NEG actual")
    axes[2].plot(
        time,
        audit["chamber_reference"][:, 0],
        "r--",
        label="Ch POS reserve guide",
    )
    axes[2].plot(
        time,
        audit["chamber_reference"][:, 1],
        "b--",
        label="Ch NEG reserve guide",
    )
    axes[2].set_ylabel("Chamber [kPa]")
    axes[2].legend(ncol=4)
    for index in range(6):
        axes[3].plot(time, audit["physical_action"][:, index], label=f"u{index + 1}")
    axes[3].set_ylabel("Physical command")
    axes[3].set_ylim(0.78, 1.02)
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(ncol=6)
    for axis in axes:
        for block in range(1, args.blocks):
            axis.axvline(block * args.block_steps / args.freq, color="black", linestyle="--", alpha=0.3)
        axis.grid(True, alpha=0.3)
    fig.suptitle("Separate PID feasibility audit (actions are not part of the training reference)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_capacity(df: pd.DataFrame, rates: dict[str, dict[str, float]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter(
        df["rate_act_pos_kpa_s"],
        df["rate_act_neg_kpa_s"],
        c=df["state_id"],
        s=9,
        alpha=0.35,
        cmap="viridis",
    )
    axis.axvline(
        rates["act_pos"]["up"], color="red", linestyle="--", label="POS target cap"
    )
    axis.axvline(-rates["act_pos"]["down"], color="red", linestyle="--")
    axis.axhline(
        rates["act_neg"]["up"], color="blue", linestyle="--", label="NEG target cap"
    )
    axis.axhline(-rates["act_neg"]["down"], color="blue", linestyle="--")
    axis.set_xlabel("dP actuator POS/dt [kPa/s]")
    axis.set_ylabel("dP actuator NEG/dt [kPa/s]")
    axis.set_title("lib3 full six-valve local signed-rate sweep")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.colorbar(scatter, ax=axis, label="operating state")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_profile(
    args: argparse.Namespace,
    time_bank: np.ndarray,
    pressure_bank: np.ndarray,
    schedules: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    capacity_summary: dict[str, Any],
    rates: dict[str, dict[str, float]],
    profile_npz: Path,
    profile_yaml: Path,
) -> None:
    # Intentionally only these two arrays: no action, valve state, q, mdot,
    # or simulator output is available to the training reference object.
    np.savez_compressed(
        profile_npz,
        time=time_bank.astype(np.float32),
        pressure=pressure_bank.astype(np.float32),
    )
    profile = {
        "kind": "optimized_trajectory_bank",
        "kwargs": {
            "trajectory_npz": str(profile_npz.relative_to(ROOT)),
            "seed": args.seed,
            "sample_mode": "coverage",
            "repeat_after_end": False,
        },
        "design": {
            "library": 3,
            "generator": "pressure-space independent POS/NEG minimum-jerk waypoints",
            "training_npz_keys": ["time", "pressure"],
            "teacher_or_oracle_action_stored_in_training_reference": False,
            "simulator_output_used_as_training_target": False,
            "simulator_and_pid_role": "offline capacity characterization and pass/fail certification only",
            "pid_certifier_gains": CERTIFIER_PID,
            "pid_certifier_common_active_min_physical": CERTIFIER_ACTIVE_MIN_PHYSICAL,
            "pid_certifier_active_min_is_policy_constraint": False,
            "action_scale": [0.8, 1.0],
            "continuous_carry": True,
            "blocks_per_hard_reset": args.blocks,
            "block_steps": args.block_steps,
            "duration_sec": float(args.blocks * args.block_steps / args.freq),
            "bank_size": int(len(pressure_bank)),
            "seed": args.seed,
            "actuator_pressure_bounds_kpa": {
                "pos": list(args.pos_bounds),
                "neg": list(args.neg_bounds),
            },
            "tracked_pressure_channels": ["act_pos", "act_neg"],
            "chamber_reserve_guide_is_training_target": False,
            "chamber_reserve_guide_role": (
                "diagnostic P_act +/- reserve margin; chamber feasibility is "
                "certified by the separate stateful closed-loop audit"
            ),
            "local_joint_support_role": (
                "simultaneous signed actuator-rate support at the nearest operating state; "
                "not an exact four-pressure trajectory certificate"
            ),
            "derived_chamber_reserve_margin_kpa": args.reserve_margin,
            "events_per_512_block_per_axis": list(args.events_per_block),
            "minimum_jerk_peak_slope": MIN_JERK_PEAK_SLOPE,
            "maximum_reference_frequency_hz": args.max_frequency_hz,
            "signed_reference_rate_limits_kpa_s": rates,
            "capacity_characterization": capacity_summary,
            "candidate_metrics": metrics,
            "schedules": schedules,
        },
    }
    with profile_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="lib3_capacity_random_medium_16x4x512_seed61098")
    parser.add_argument("--seed", type=int, default=61098)
    parser.add_argument("--bank-size", type=int, default=16)
    parser.add_argument("--candidates-per-member", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--pos-bounds", type=parse_float_pair, default=(110.0, 145.0))
    parser.add_argument("--neg-bounds", type=parse_float_pair, default=(58.0, 82.0))
    parser.add_argument("--startup-pos", type=parse_float_pair, default=(120.0, 130.0))
    parser.add_argument("--startup-neg", type=parse_float_pair, default=(68.0, 76.0))
    parser.add_argument("--pos-delta", type=parse_float_pair, default=(6.0, 15.0))
    parser.add_argument("--neg-delta", type=parse_float_pair, default=(3.0, 7.0))
    parser.add_argument("--events-per-block", type=parse_int_pair, default=(3, 4))
    parser.add_argument("--hold-sec", type=parse_float_pair, default=(0.35, 1.65))
    parser.add_argument("--min-transition", type=float, default=0.8)
    parser.add_argument("--max-frequency-hz", type=float, default=0.35)
    parser.add_argument("--reserve-margin", type=float, default=35.0)
    parser.add_argument("--startup-pos-rate", type=float, default=12.0)
    parser.add_argument("--startup-neg-rate", type=float, default=24.0)
    parser.add_argument("--capacity-grid", type=int, default=3)
    parser.add_argument(
        "--reuse-capacity",
        action="store_true",
        help="reuse output-dir/capacity_sweep.csv instead of rerunning the full local sweep",
    )
    parser.add_argument("--capacity-warmup-sec", type=float, default=0.2)
    parser.add_argument("--capacity-measure-sec", type=float, default=0.4)
    parser.add_argument("--capacity-quantile", type=float, default=0.90)
    parser.add_argument("--capacity-utilization", type=float, default=0.45)
    parser.add_argument("--pos-rate-up-cap", type=float, default=10.0)
    parser.add_argument("--pos-rate-down-cap", type=float, default=12.0)
    parser.add_argument("--neg-rate-up-cap", type=float, default=4.5)
    parser.add_argument("--neg-rate-down-cap", type=float, default=5.0)
    parser.add_argument("--min-joint-support", type=float, default=0.98)
    parser.add_argument("--max-pos-rmse", type=float, default=8.0)
    parser.add_argument("--max-neg-rmse", type=float, default=5.5)
    parser.add_argument("--min-pos-direction", type=float, default=0.65)
    parser.add_argument("--min-neg-direction", type=float, default=0.65)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pneu_rl/exp/lib3_capacity_random_reference"),
    )
    args = parser.parse_args()
    if args.bank_size <= 0 or args.candidates_per_member <= 0:
        parser.error("bank size and candidate count must be positive")
    if args.blocks <= 0 or args.block_steps <= 0:
        parser.error("blocks and block steps must be positive")
    if args.capacity_grid < 2:
        parser.error("capacity grid must have at least two points")
    if not 0.0 < args.capacity_quantile < 1.0:
        parser.error("capacity quantile must lie in (0, 1)")
    if not 0.0 < args.capacity_utilization <= 1.0:
        parser.error("capacity utilization must lie in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_npz = ROOT / "pneu_rl" / "config" / "reference" / f"{args.name}.npz"
    profile_yaml = ROOT / "pneu_rl" / "config" / "reference" / f"{args.name}.yaml"
    image_path = ROOT / "pneu_rl" / "img" / "reference" / f"{args.name}.png"
    profile_npz.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    states = operating_states(args.pos_bounds, args.neg_bounds, args.reserve_margin)
    capacity_path = output_dir / "capacity_sweep.csv"
    if args.reuse_capacity and capacity_path.is_file():
        print(f"reusing capacity sweep: {capacity_path}")
        capacity = pd.read_csv(capacity_path)
    else:
        print(f"capacity sweep: {len(states)} states x {args.capacity_grid ** 6} actions")
        capacity = run_capacity_sweep(
            states,
            grid_points=args.capacity_grid,
            warmup_sec=args.capacity_warmup_sec,
            measure_sec=args.capacity_measure_sec,
        )
        capacity.to_csv(capacity_path, index=False)
    capacity_summary = summarize_capacity(capacity, args.capacity_quantile)
    caps = {
        "act_pos": {"up": args.pos_rate_up_cap, "down": args.pos_rate_down_cap},
        "act_neg": {"up": args.neg_rate_up_cap, "down": args.neg_rate_down_cap},
    }
    rates = derive_reference_rates(
        capacity_summary,
        utilization=args.capacity_utilization,
        caps=caps,
    )
    capacity_summary["reference_rate_derivation"] = {
        "utilization_of_q20_capacity": args.capacity_utilization,
        "manual_upper_caps_kpa_s": caps,
        "resolved_reference_rates_kpa_s": rates,
    }
    with (output_dir / "capacity_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(capacity_summary, handle, sort_keys=False)
    plot_capacity(capacity, rates, output_dir / "capacity_map.png")
    print("resolved signed reference rates:")
    print(yaml.safe_dump(rates, sort_keys=False))

    capacity_states, capacity_clouds = nearest_capacity_groups(capacity)
    master_rng = random.Random(args.seed)
    accepted: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []
    for member in range(args.bank_size):
        best: dict[str, Any] | None = None
        for candidate_index in range(args.candidates_per_member):
            candidate_seed = master_rng.randrange(1, 2**31 - 1)
            rng = random.Random(candidate_seed)
            time, pressure, schedule = make_pressure_candidate(rng, args, rates)
            chamber_reference = derive_chamber_reference(pressure, args.reserve_margin)
            support = joint_rate_support(
                time,
                pressure,
                chamber_reference,
                capacity_states,
                capacity_clouds,
            )
            audit = run_pid_certifier(time, pressure, margin=args.reserve_margin)
            metrics = candidate_metrics(time, pressure, audit, support, args)
            row = {
                "member": member,
                "candidate": candidate_index,
                "seed": candidate_seed,
                "passed": metrics["passed"],
                "score": metrics["score"],
                **metrics["reference"],
                **metrics["pid_certifier"],
            }
            all_candidate_rows.append(row)
            candidate = {
                "time": time,
                "pressure": pressure,
                "schedule": schedule,
                "audit": audit,
                "metrics": metrics,
                "seed": candidate_seed,
            }
            if metrics["passed"] and (best is None or metrics["score"] < best["metrics"]["score"]):
                best = candidate
        if best is None:
            ranked = sorted(
                [row for row in all_candidate_rows if row["member"] == member],
                key=lambda row: row["score"],
            )
            pd.DataFrame(all_candidate_rows).to_csv(output_dir / "candidate_screening.csv", index=False)
            raise RuntimeError(
                f"no certified candidate for bank member {member}; best={json.dumps(ranked[0], default=float)}"
            )
        accepted.append(best)
        pid = best["metrics"]["pid_certifier"]
        print(
            f"member {member + 1:02d}/{args.bank_size}: "
            f"seed={best['seed']} score={best['metrics']['score']:.3f} "
            f"RMSE=({pid['pos_rmse_kpa']:.2f}, {pid['neg_rmse_kpa']:.2f}) "
            f"direction=({pid['pos_direction_fraction']:.2f}, {pid['neg_direction_fraction']:.2f})"
        )

    pd.DataFrame(all_candidate_rows).to_csv(output_dir / "candidate_screening.csv", index=False)
    time_bank = np.stack([item["time"] for item in accepted])
    pressure_bank = np.stack([item["pressure"] for item in accepted])
    schedules = [item["schedule"] for item in accepted]
    metrics = [item["metrics"] for item in accepted]
    save_profile(
        args,
        time_bank,
        pressure_bank,
        schedules,
        metrics,
        capacity_summary,
        rates,
        profile_npz,
        profile_yaml,
    )

    audit_path = output_dir / "pid_certification_audit.npz"
    np.savez_compressed(
        audit_path,
        time=time_bank.astype(np.float32),
        reference=pressure_bank.astype(np.float32),
        pressure=np.stack([item["audit"]["pressure"] for item in accepted]).astype(np.float32),
        chamber_reference=np.stack(
            [item["audit"]["chamber_reference"] for item in accepted]
        ).astype(np.float32),
        raw_action=np.stack([item["audit"]["raw_action"] for item in accepted]).astype(np.float32),
        physical_action=np.stack(
            [item["audit"]["physical_action"] for item in accepted]
        ).astype(np.float32),
    )
    plot_reference(
        accepted[0]["time"],
        accepted[0]["pressure"],
        accepted[0]["schedule"],
        rates,
        image_path,
        args,
    )
    plot_bank_overview(
        time_bank,
        pressure_bank,
        image_path.with_name(f"{args.name}_bank.png"),
        args,
    )
    plot_pid_audit(
        accepted[0]["time"],
        accepted[0]["pressure"],
        accepted[0]["audit"],
        output_dir / "pid_audit_member01.png",
        args,
    )
    print(f"training profile: {profile_yaml.relative_to(ROOT)}")
    print(f"training target NPZ (time+pressure only): {profile_npz.relative_to(ROOT)}")
    print(f"reference plot: {image_path.relative_to(ROOT)}")
    print(f"separate PID/action audit: {audit_path.resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
