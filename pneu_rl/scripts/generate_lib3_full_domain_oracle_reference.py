#!/usr/bin/env python3
"""Generate a hard lib3 reference by replaying smooth random valve commands.

The reference is the actuator-pressure output of the complete lib3 C++
simulator.  It is not drawn in pressure space and no PID/controller tracks a
pressure target during generation.  Candidate six-valve command programmes
are sampled over the same physical command domain, replayed from atmosphere,
and accepted only after pressure/rate/spectral checks.

The saved oracle command is for reproducibility only.  It is never exposed to
the SAC policy by ``OptimizedTrajectoryBankRef``.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_utils" / "src",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.sim3 import PneuSim


ATM = 101.325
VALVE_NAMES = (
    "ch_pos",
    "ch_neg",
    "act_pos_in",
    "act_pos_out",
    "act_neg_in",
    "act_neg_out",
)
PRESSURE_NAMES = ("ch_pos", "ch_neg", "act_pos", "act_neg")

# A family determines which actuator span is required/scored in one block,
# not which valve is disabled or where a valve is allowed to move.  All six
# coordinates stay active and every valve uses the same COMMAND_DOMAIN.  This
# matters because an apparently "inactive" branch can drift for 10.24 s and
# invalidate an otherwise useful candidate.
FAMILY_ACTIVE = {
    "coupled": (0, 1, 2, 3, 4, 5),
    "pos_branch": (0, 1, 2, 3, 4, 5),
    "neg_branch": (0, 1, 2, 3, 4, 5),
    "mixed": (0, 1, 2, 3, 4, 5),
}
FAMILY_ORDER = ("coupled", "pos_branch", "neg_branch", "mixed")


def parse_pair(text: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 2 or values[0] > values[1]:
        raise argparse.ArgumentTypeError("expected an ascending comma-separated pair")
    return values[0], values[1]


def quintic(progress: np.ndarray) -> np.ndarray:
    tau = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def allocate_durations(
    rng: random.Random,
    count: int,
    duration: float,
    transition_range: tuple[float, float],
    hold_range: tuple[float, float],
) -> tuple[list[float], list[float]]:
    minimum = count * (transition_range[0] + hold_range[0])
    capacities = [transition_range[1] - transition_range[0]] * count
    capacities += [hold_range[1] - hold_range[0]] * count
    if minimum > duration + 1e-9 or minimum + sum(capacities) < duration - 1e-9:
        raise ValueError("transition/hold ranges cannot fill the requested duration")
    remaining = duration - minimum
    increments = [0.0] * len(capacities)
    order = list(range(len(capacities)))
    rng.shuffle(order)
    future_capacity = float(sum(capacities))
    for position, index in enumerate(order):
        capacity = capacities[index]
        future_capacity -= capacity
        lower = max(0.0, remaining - future_capacity)
        upper = min(capacity, remaining)
        increment = upper if position == len(order) - 1 else rng.uniform(lower, upper)
        increments[index] = increment
        remaining -= increment
    transitions = [transition_range[0] + increments[index] for index in range(count)]
    holds = [hold_range[0] + increments[count + index] for index in range(count)]
    return transitions, holds


def sample_full_domain_target(
    rng: random.Random,
    current: float,
    command_domain: tuple[float, float],
    min_delta: float,
    full_domain_probability: float,
    endpoint_probability: float,
    local_sigma: float,
) -> float:
    """Sample one target without valve-specific command bounds.

    Local proposals make feasible candidate discovery efficient; the uniform
    and endpoint mixture keeps every value in the common physical domain
    reachable.  This changes only the offline proposal distribution, never
    the policy action space.
    """
    lower, upper = command_domain
    for _ in range(64):
        draw = rng.random()
        if draw < endpoint_probability:
            target = lower
        elif draw < 2.0 * endpoint_probability:
            target = upper
        elif draw < 2.0 * endpoint_probability + full_domain_probability:
            target = rng.uniform(lower, upper)
        else:
            target = min(max(rng.gauss(current, local_sigma), lower), upper)
        if abs(target - current) >= min_delta:
            return float(target)
    return float(lower if current > 0.5 * (lower + upper) else upper)


def axis_program(
    time: np.ndarray,
    rng: random.Random,
    initial: float,
    *,
    active: bool,
    initial_hold_sec: float,
    command_domain: tuple[float, float],
    count_range: tuple[int, int],
    transition_range: tuple[float, float],
    hold_range: tuple[float, float],
    min_delta: float,
    full_domain_probability: float,
    endpoint_probability: float,
    local_sigma: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    values = np.full(len(time), float(initial), dtype=np.float64)
    if not active:
        return values, []

    available_duration = float(time[-1]) - float(initial_hold_sec)
    possible = [
        count
        for count in range(count_range[0], count_range[1] + 1)
        if count * (transition_range[0] + hold_range[0]) <= available_duration + 1e-9
        and count * (transition_range[1] + hold_range[1]) >= available_duration - 1e-9
    ]
    if not possible:
        raise ValueError("no transition count can fill one 512-step block")
    count = rng.choice(possible)
    transitions, holds = allocate_durations(
        rng, count, available_duration, transition_range, hold_range
    )
    cursor = float(initial_hold_sec)
    current = float(initial)
    segments: list[dict[str, float]] = []
    for transition, hold in zip(transitions, holds):
        target = sample_full_domain_target(
            rng,
            current,
            command_domain,
            min_delta,
            full_domain_probability,
            endpoint_probability,
            local_sigma,
        )
        end = cursor + transition
        mask = (time >= cursor) & (time < end)
        if np.any(mask):
            values[mask] = current + (target - current) * quintic(
                (time[mask] - cursor) / transition
            )
        values[time >= end] = target
        segments.append(
            {
                "start_sec": float(cursor),
                "transition_sec": float(transition),
                "hold_sec": float(hold),
                "physical_start": float(current),
                "physical_target": float(target),
                "end_sec": float(end + hold),
            }
        )
        cursor = end + hold
        current = target
    return values, segments


def propose_block(
    rng: random.Random,
    previous_action: np.ndarray,
    family: str,
    args: argparse.Namespace,
    *,
    initial_hold_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    block_time = np.arange(args.horizon_steps + 1, dtype=np.float64) / args.freq
    active = set(FAMILY_ACTIVE[family])
    columns = []
    valve_segments: dict[str, list[dict[str, float]]] = {}
    for index, valve_name in enumerate(VALVE_NAMES):
        values, segments = axis_program(
            block_time,
            rng,
            float(previous_action[index]),
            active=index in active,
            initial_hold_sec=initial_hold_sec,
            command_domain=args.command_domain,
            count_range=args.transition_count,
            transition_range=args.transition_sec,
            hold_range=args.hold_sec,
            min_delta=args.min_command_delta,
            full_domain_probability=args.full_domain_probability,
            endpoint_probability=args.endpoint_probability,
            local_sigma=args.local_sigma,
        )
        columns.append(values)
        valve_segments[valve_name] = segments
    return np.column_stack(columns), {
        "family": family,
        "active_valves": [VALVE_NAMES[index] for index in sorted(active)],
        "segments": valve_segments,
    }


def propose_recovery_block(
    previous_action: np.ndarray,
    family: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add one controller-free joint recovery candidate to every pool.

    Greedy block selection can otherwise leave a hidden valve/chamber state
    from which every random continuation violates a safety band.  This is a
    fixed open-loop minimum-jerk transition to the measured neutral command,
    not pressure feedback and not an action restriction.
    """
    block_time = np.arange(args.horizon_steps + 1, dtype=np.float64) / args.freq
    transition = min(2.0, float(block_time[-1]))
    blend = quintic(block_time / transition)
    target = float(args.initial_physical_command)
    values = previous_action[None, :] + (
        target - previous_action[None, :]
    ) * blend[:, None]
    return values, {
        "family": family,
        "proposal_kind": "open_loop_neutral_recovery",
        "active_valves": list(VALVE_NAMES),
        "segments": {
            name: [
                {
                    "start_sec": 0.0,
                    "transition_sec": transition,
                    "hold_sec": float(block_time[-1] - transition),
                    "physical_start": float(previous_action[index]),
                    "physical_target": target,
                    "end_sec": float(block_time[-1]),
                }
            ]
            for index, name in enumerate(VALVE_NAMES)
        },
    }


def propose_physics_excitation_block(
    rng: random.Random,
    previous_action: np.ndarray,
    family: str,
    args: argparse.Namespace,
    *,
    initial_hold_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample simultaneous branch excitation using the known flow topology.

    Positive drive couples ``u1 down, u3 up, u4 down``; negative drive
    couples ``u2 up, u5 up, u6 down``.  Drive signs and POS/NEG timing are
    independently random.  This is open-loop task generation, not a pressure
    controller, and it does not constrain the learned six-dimensional action.
    """
    time = np.arange(args.horizon_steps + 1, dtype=np.float64) / args.freq
    lower, upper = args.command_domain
    previous = np.asarray(previous_action, dtype=np.float64)

    def branch_program(
        indices: tuple[int, int, int],
        rise_sign: np.ndarray,
        lower_sign: np.ndarray,
        count_range: tuple[int, int],
        initial_direction: float | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        available = float(time[-1]) - initial_hold_sec
        possible = [
            count
            for count in range(count_range[0], count_range[1] + 1)
            if count * (max(1.2, args.transition_sec[0]) + args.hold_sec[0])
            <= available + 1e-9
            and count * (args.transition_sec[1] + args.hold_sec[1])
            >= available - 1e-9
        ]
        count = rng.choice(possible)
        transitions, holds = allocate_durations(
            rng,
            count,
            available,
            (max(1.2, args.transition_sec[0]), args.transition_sec[1]),
            args.hold_sec,
        )
        branch = np.tile(previous[list(indices)], (len(time), 1))
        current = previous[list(indices)].copy()
        cursor = float(initial_hold_sec)
        segments: list[dict[str, Any]] = []
        direction = (
            rng.choice((-1.0, 1.0))
            if initial_direction is None
            else float(initial_direction)
        )
        for transition, hold in zip(transitions, holds):
            # Most tasks are local enough to preserve the accumulated chamber
            # state.  A smaller full-domain component can still drive any of
            # the three commands all the way to the common 0.8/1.0 boundary.
            if rng.random() < 0.20:
                magnitude = rng.uniform(0.04, upper - lower)
            else:
                magnitude = rng.uniform(0.012, 0.05)
            delta_sign = rise_sign if direction > 0.0 else lower_sign
            target = np.clip(current + delta_sign * magnitude, lower, upper)
            end = cursor + transition
            mask = (time >= cursor) & (time < end)
            if np.any(mask):
                blend = quintic((time[mask] - cursor) / transition)[:, None]
                branch[mask] = current + (target - current) * blend
            branch[time >= end] = target
            segments.append(
                {
                    "start_sec": float(cursor),
                    "transition_sec": float(transition),
                    "hold_sec": float(hold),
                    "direction": "raise" if direction > 0.0 else "lower",
                    "magnitude": float(magnitude),
                    "physical_start": current.tolist(),
                    "physical_target": target.tolist(),
                    "end_sec": float(end + hold),
                }
            )
            cursor = end + hold
            current = target
            direction *= -1.0
        return branch, segments

    pos_count = (2, 3) if family in {"pos_branch", "coupled", "mixed"} else (1, 2)
    neg_count = (2, 3) if family in {"neg_branch", "coupled", "mixed"} else (1, 2)
    pos_values, pos_segments = branch_program(
        (0, 2, 3),
        np.asarray([-1.0, 1.0, -1.0]),
        np.asarray([1.0, -1.0, 1.0]),
        pos_count,
    )
    neg_values, neg_segments = branch_program(
        (1, 4, 5),
        np.asarray([1.0, 1.0, -1.0]),
        # While evacuating the actuator, keep the chamber valve from driving
        # the NEG reservoir below its safety band: u5 closes, u6 opens, and
        # u2 supplies reserve instead of being forced downward with u5.
        np.asarray([1.0, -1.0, 1.0]),
        neg_count,
        initial_direction=(-1.0 if family == "neg_branch" else None),
    )
    values = np.tile(previous, (len(time), 1))
    values[:, (0, 2, 3)] = pos_values
    values[:, (1, 4, 5)] = neg_values
    return values, {
        "family": family,
        "proposal_kind": "physics_topology_open_loop",
        "active_valves": list(VALVE_NAMES),
        "positive_branch_segments": pos_segments,
        "negative_branch_segments": neg_segments,
        "positive_negative_timing_independent": True,
    }


def simulate(
    physical_action: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay one complete command prefix from atmosphere."""
    sim = getattr(args, "_shared_sim", None)
    if sim is None:
        sim = PneuSim(
            freq=args.freq,
            delay=0.0,
            init_pos_press=ATM,
            init_neg_press=ATM,
            init_act_pos_press=ATM,
            init_act_neg_press=ATM,
            noise=False,
            scale=False,
        )
        args._shared_sim = sim
    sim.set_init_press(ATM, ATM, ATM, ATM)
    pressure = np.full((len(physical_action), 4), np.nan, dtype=np.float64)
    q_pred = np.full((len(physical_action), 6), np.nan, dtype=np.float64)
    valve_z = np.full((len(physical_action), 6), np.nan, dtype=np.float64)
    mdot = np.full((len(physical_action), 6), np.nan, dtype=np.float64)
    pressure[0] = ATM
    q_pred[0] = 0.0
    valve_z[0] = 0.0
    mdot[0] = 0.0
    for index in range(1, len(physical_action)):
        observation, _ = sim.observe(physical_action[index])
        pressure[index] = observation[1:5]
        debug = sim.get_valve_debug()
        for valve_index, valve_name in enumerate(VALVE_NAMES):
            q_pred[index, valve_index] = debug[f"{valve_name}_q_pred_lpm"]
            valve_z[index, valve_index] = debug[f"{valve_name}_z"]
            mdot[index, valve_index] = debug[f"{valve_name}_mdot"]
        if not np.all(np.isfinite(pressure[index])):
            break
        if any(
            not (lower <= pressure[index, channel] <= upper)
            for channel, (lower, upper) in enumerate(args.emergency_bands)
        ):
            break
    return pressure, q_pred, valve_z, mdot


def high_frequency_fraction(values: np.ndarray, freq: float, cutoff_hz: float) -> float:
    signal = np.asarray(values, dtype=np.float64)
    if len(signal) < 8 or not np.all(np.isfinite(signal)):
        return math.inf
    # A power *fraction* is ill-conditioned for an almost constant pressure:
    # tiny numerical/valve residuals can account for most of its tiny total
    # power while being physically irrelevant.
    if float(np.ptp(signal)) < 1.0:
        return 0.0
    signal = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / freq)
    spectrum[0] = 0.0
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(spectrum[frequencies >= cutoff_hz]) / total)


def feature_bins(
    pressure: np.ndarray,
    rate: np.ndarray,
    args: argparse.Namespace,
) -> set[tuple[int, int, int, int]]:
    act = pressure[:, 2:4]
    pos = np.clip(
        (act[:, 0] - args.act_pos_band[0])
        / (args.act_pos_band[1] - args.act_pos_band[0]),
        0.0,
        0.999,
    )
    neg = np.clip(
        (act[:, 1] - args.act_neg_band[0])
        / (args.act_neg_band[1] - args.act_neg_band[0]),
        0.0,
        0.999,
    )
    pos_rate = np.digitize(rate[:, 2], [-4.0, 0.0, 4.0])
    neg_rate = np.digitize(rate[:, 3], [-2.0, 0.0, 2.0])
    return {
        (int(p * 7), int(n * 7), int(dp), int(dn))
        for p, n, dp, dn in zip(pos, neg, pos_rate, neg_rate)
    }


def evaluate_candidate(
    pressure: np.ndarray,
    physical_action: np.ndarray,
    start: int,
    visited: set[tuple[int, int, int, int]],
    family: str,
    args: argparse.Namespace,
    *,
    allow_low_activity: bool = False,
) -> tuple[float, dict[str, Any]] | tuple[None, str]:
    checked_start = start
    if start == 0:
        checked_start += int(round(args.ignore_initial_sec * args.freq))
    block = pressure[checked_start:]
    if len(block) < 8 or not np.all(np.isfinite(block)):
        return None, "nonfinite_or_emergency"
    rate_all = np.gradient(pressure, 1.0 / args.freq, axis=0)
    rate = rate_all[checked_start:]
    hard_bands = (
        args.ch_pos_band,
        args.ch_neg_band,
        args.act_pos_band,
        args.act_neg_band,
    )
    for channel, band in enumerate(hard_bands):
        if np.any(block[:, channel] < band[0]) or np.any(block[:, channel] > band[1]):
            return None, f"{PRESSURE_NAMES[channel]}_band"
    rate_limits = (
        args.ch_pos_rate_max,
        args.ch_neg_rate_max,
        args.act_pos_rate_max,
        args.act_neg_rate_max,
    )
    for channel, limit in enumerate(rate_limits):
        if np.max(np.abs(rate[:, channel])) > limit:
            return None, f"{PRESSURE_NAMES[channel]}_rate"

    span = np.ptp(block, axis=0)
    if not allow_low_activity:
        if family == "pos_branch" and span[2] < args.min_pos_span:
            return None, "act_pos_span"
        if family == "neg_branch" and span[3] < args.min_neg_span:
            return None, "act_neg_span"
        if family in {"coupled", "mixed"}:
            if span[2] < 0.7 * args.min_pos_span:
                return None, "act_pos_span"
            if span[3] < 0.7 * args.min_neg_span:
                return None, "act_neg_span"

    hf = np.asarray(
        [
            high_frequency_fraction(block[:, 2], args.freq, args.hf_cutoff_hz),
            high_frequency_fraction(block[:, 3], args.freq, args.hf_cutoff_hz),
        ]
    )
    if hf[0] > args.pos_hf_fraction_max:
        return None, "act_pos_high_frequency"
    if hf[1] > args.neg_hf_fraction_max:
        return None, "act_neg_high_frequency"

    terminal_bands = (
        args.terminal_ch_pos,
        args.terminal_ch_neg,
        args.terminal_act_pos,
        args.terminal_act_neg,
    )
    endpoint = block[-1]
    for channel, band in enumerate(terminal_bands):
        if not (band[0] <= endpoint[channel] <= band[1]):
            return None, f"{PRESSURE_NAMES[channel]}_terminal"

    bins = feature_bins(block, rate, args)
    coverage_gain = len(bins - visited)
    mean_abs_rate = np.mean(np.abs(rate[:, 2:4]), axis=0)
    action_block = physical_action[start:]
    action_span = np.ptp(action_block, axis=0)
    score = (
        3.0 * coverage_gain
        + 2.0 * span[2] / max(args.act_pos_band[1] - args.act_pos_band[0], 1.0)
        + 2.0 * span[3] / max(args.act_neg_band[1] - args.act_neg_band[0], 1.0)
        + 0.5 * mean_abs_rate[0] / args.act_pos_rate_max
        + 0.5 * mean_abs_rate[1] / args.act_neg_rate_max
        + 0.5 * float(np.sum(action_span))
        - 2.0 * float(np.sum(hf))
    )
    metadata = {
        "score": float(score),
        "coverage_gain": int(coverage_gain),
        "pressure_min_kpa": block.min(axis=0).tolist(),
        "pressure_max_kpa": block.max(axis=0).tolist(),
        "pressure_span_kpa": span.tolist(),
        "rate_abs_max_kpa_s": np.max(np.abs(rate), axis=0).tolist(),
        "rate_mean_abs_actuator_kpa_s": mean_abs_rate.tolist(),
        "actuator_hf_power_fraction": hf.tolist(),
        "physical_action_min": action_block.min(axis=0).tolist(),
        "physical_action_max": action_block.max(axis=0).tolist(),
    }
    return float(score), metadata


def generate_one(
    args: argparse.Namespace,
    *,
    trajectory_index: int,
    visited: set[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = random.Random(args.seed + 10_007 * trajectory_index)
    committed = np.full((1, 6), args.initial_physical_command, dtype=np.float64)
    selected_pressure = np.full((1, 4), ATM, dtype=np.float64)
    selected_q = np.zeros((1, 6), dtype=np.float64)
    selected_z = np.zeros((1, 6), dtype=np.float64)
    selected_mdot = np.zeros((1, 6), dtype=np.float64)
    schedule: list[dict[str, Any]] = []

    for block_index in range(args.blocks):
        family = FAMILY_ORDER[block_index % len(FAMILY_ORDER)]
        start = len(committed) - 1
        best: tuple[
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, Any],
            dict[str, Any],
        ] | None = None
        rejected: Counter[str] = Counter()
        for candidate_index in range(args.candidates_per_block):
            if candidate_index == 0:
                proposal, proposal_meta = propose_recovery_block(
                    committed[-1], family, args
                )
            elif candidate_index % 2 == 1:
                proposal, proposal_meta = propose_physics_excitation_block(
                    rng,
                    committed[-1],
                    family,
                    args,
                    initial_hold_sec=(args.initial_hold_sec if block_index == 0 else 0.0),
                )
            else:
                proposal, proposal_meta = propose_block(
                    rng,
                    committed[-1],
                    family,
                    args,
                    initial_hold_sec=(args.initial_hold_sec if block_index == 0 else 0.0),
                )
            candidate_action = np.vstack((committed, proposal[1:]))
            pressure, q_pred, valve_z, mdot = simulate(candidate_action, args)
            result, detail = evaluate_candidate(
                pressure,
                candidate_action,
                start,
                visited,
                family,
                args,
                allow_low_activity=(
                    proposal_meta.get("proposal_kind") == "open_loop_neutral_recovery"
                ),
            )
            if result is None:
                rejected[str(detail)] += 1
                continue
            score = float(result)
            if best is None or score > best[0]:
                best = (
                    score,
                    candidate_action,
                    pressure,
                    q_pred,
                    valve_z,
                    mdot,
                    proposal_meta,
                    detail,
                )
        if best is None:
            raise RuntimeError(
                f"trajectory {trajectory_index} block {block_index}: no feasible candidate; "
                f"rejections={dict(rejected)}"
            )
        (
            _,
            committed,
            selected_pressure,
            selected_q,
            selected_z,
            selected_mdot,
            proposal_meta,
            score_meta,
        ) = best
        block_start = len(committed) - args.horizon_steps - 1
        block_rate = np.gradient(selected_pressure, 1.0 / args.freq, axis=0)[block_start:]
        visited.update(feature_bins(selected_pressure[block_start:], block_rate, args))
        schedule.append(
            {
                "trajectory": trajectory_index,
                "block": block_index,
                "start_sec": float(block_index * args.horizon_steps / args.freq),
                "candidate_count": args.candidates_per_block,
                "rejections": dict(rejected),
                **proposal_meta,
                **score_meta,
            }
        )
        print(
            f"[ INFO] oracle-ref | trajectory {trajectory_index + 1}/{args.bank_size} "
            f"| block {block_index + 1}/{args.blocks} | {family} "
            f"| act=({selected_pressure[-1, 2]:.2f}, {selected_pressure[-1, 3]:.2f}) "
            f"| coverage={len(visited)}"
        )
    return (
        committed,
        selected_pressure,
        selected_q,
        selected_z,
        selected_mdot,
        schedule,
    )


def trajectory_metrics(
    pressure: np.ndarray,
    physical_action: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rate = np.gradient(pressure, 1.0 / args.freq, axis=0)
    start = min(len(pressure) - 1, int(round(args.ignore_initial_sec * args.freq)))
    checked = pressure[start:]
    checked_rate = rate[start:]
    return {
        "pressure_min_after_startup_kpa": checked.min(axis=0).tolist(),
        "pressure_max_after_startup_kpa": checked.max(axis=0).tolist(),
        "pressure_span_after_startup_kpa": np.ptp(checked, axis=0).tolist(),
        "rate_abs_max_after_startup_kpa_s": np.max(np.abs(checked_rate), axis=0).tolist(),
        "actuator_hf_power_fraction_after_startup": [
            high_frequency_fraction(checked[:, 2], args.freq, args.hf_cutoff_hz),
            high_frequency_fraction(checked[:, 3], args.freq, args.hf_cutoff_hz),
        ],
        "physical_action_min": physical_action.min(axis=0).tolist(),
        "physical_action_max": physical_action.max(axis=0).tolist(),
    }


def plot_reference(
    time: np.ndarray,
    pressure: np.ndarray,
    physical_action: np.ndarray,
    q_pred: np.ndarray,
    args: argparse.Namespace,
) -> None:
    rate = np.gradient(pressure, 1.0 / args.freq, axis=0)
    fig, axes = plt.subplots(5, 1, figsize=(17, 15), sharex=True)
    axes[0].plot(time, pressure[:, 2], color="red", label="ACT POS ref")
    axes[0].plot(time, pressure[:, 3], color="blue", label="ACT NEG ref")
    axes[0].set_ylabel("Actuator [kPa]")

    axes[1].plot(time, pressure[:, 0], color="darkred", label="CH POS (oracle state)")
    axes[1].plot(time, pressure[:, 1], color="navy", label="CH NEG (oracle state)")
    axes[1].plot(time, pressure[:, 2], "r--", alpha=0.65, label="ACT POS")
    axes[1].plot(time, pressure[:, 3], "b--", alpha=0.65, label="ACT NEG")
    axes[1].set_ylabel("All pressure [kPa]")

    for channel, color, label in (
        (0, "darkred", "dCH POS/dt"),
        (1, "navy", "dCH NEG/dt"),
        (2, "red", "dACT POS/dt"),
        (3, "blue", "dACT NEG/dt"),
    ):
        axes[2].plot(time, rate[:, channel], color=color, linewidth=1.0, label=label)
    axes[2].set_ylabel("Rate [kPa/s]")

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    for index, (name, color) in enumerate(zip(VALVE_NAMES, colors)):
        axes[3].plot(time, physical_action[:, index], color=color, label=f"u{index + 1} {name}")
    axes[3].set_ylim(args.command_domain[0] - 0.01, args.command_domain[1] + 0.01)
    axes[3].set_ylabel("Physical command")

    for index, (name, color) in enumerate(zip(VALVE_NAMES, colors)):
        axes[4].plot(time, q_pred[:, index], color=color, label=f"u{index + 1} {name}")
    axes[4].set_ylabel("q_pred [L/min]")
    axes[4].set_xlabel("Continuous time [s]")

    boundary_sec = args.horizon_steps / args.freq
    for axis in axes:
        for block in range(1, args.blocks):
            axis.axvline(block * boundary_sec, color="black", linestyle=":", linewidth=0.9)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=3, fontsize=8)
    fig.suptitle(
        "lib3 full-domain simulator-oracle reference | "
        "pressure is plant output; physical commands are not given to RL"
    )
    fig.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bank(
    args: argparse.Namespace,
    actions: list[np.ndarray],
    pressures: list[np.ndarray],
    q_values: list[np.ndarray],
    z_values: list[np.ndarray],
    mdot_values: list[np.ndarray],
    schedules: list[list[dict[str, Any]]],
) -> None:
    action_bank = np.stack(actions)
    pressure_bank = np.stack(pressures)
    q_bank = np.stack(q_values)
    z_bank = np.stack(z_values)
    mdot_bank = np.stack(mdot_values)
    normalized_action = (action_bank - args.command_domain[0]) / (
        args.command_domain[1] - args.command_domain[0]
    )
    time = np.arange(action_bank.shape[1], dtype=np.float64) / args.freq
    time_bank = np.repeat(time[None, :], args.bank_size, axis=0)

    replay_rmse = []
    replay_max_abs = []
    stored_replay_rmse = []
    stored_replay_max_abs = []
    for index in range(args.bank_size):
        replay_pressure, _, _, _ = simulate(action_bank[index], args)
        error = replay_pressure - pressure_bank[index]
        replay_rmse.append(np.sqrt(np.mean(error**2, axis=0)).tolist())
        replay_max_abs.append(np.max(np.abs(error), axis=0).tolist())
        stored_action = action_bank[index].astype(np.float32).astype(np.float64)
        stored_pressure = pressure_bank[index].astype(np.float32).astype(np.float64)
        stored_replay_pressure, _, _, _ = simulate(stored_action, args)
        stored_error = stored_replay_pressure - stored_pressure
        stored_replay_rmse.append(
            np.sqrt(np.mean(stored_error**2, axis=0)).tolist()
        )
        stored_replay_max_abs.append(
            np.max(np.abs(stored_error), axis=0).tolist()
        )

    metrics = [
        trajectory_metrics(pressure_bank[index], action_bank[index], args)
        for index in range(args.bank_size)
    ]
    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        time=time_bank.astype(np.float32),
        pressure=pressure_bank[:, :, 2:4].astype(np.float32),
        full_pressure=pressure_bank.astype(np.float32),
        physical_action=action_bank.astype(np.float32),
        normalized_action=normalized_action.astype(np.float32),
        raw_action=normalized_action.astype(np.float32),
        q_pred_lpm=q_bank.astype(np.float32),
        valve_z=z_bank.astype(np.float32),
        mdot=mdot_bank.astype(np.float32),
    )

    profile = {
        "kind": "optimized_trajectory_bank",
        "kwargs": {
            "trajectory_npz": str(args.npz_output),
            "seed": args.seed,
            "sample_mode": "coverage",
            "repeat_after_end": False,
        },
        "design": {
            "library": 3,
            "continuous_carry": True,
            "bank_size": args.bank_size,
            "duration_sec": float(time[-1]),
            "block_steps": args.horizon_steps,
            "blocks_per_hard_reset": args.blocks,
            "action_scale": list(args.command_domain),
            "common_physical_command_domain_all_six_valves": list(args.command_domain),
            "valve_specific_command_bounds_used": False,
            "policy_action_projection_used": False,
            "reference_pressure_channels": ["act_pos", "act_neg"],
            "stored_diagnostic_pressure_channels": list(PRESSURE_NAMES),
            "generator": (
                "smooth random six-valve command programmes selected by complete "
                "lib3 simulator replay"
            ),
            "initial_pressure_kpa": [ATM, ATM, ATM, ATM],
            "initial_physical_command": [args.initial_physical_command] * 6,
            "candidate_filters": {
                "pressure_kpa": {
                    "ch_pos": list(args.ch_pos_band),
                    "ch_neg": list(args.ch_neg_band),
                    "act_pos": list(args.act_pos_band),
                    "act_neg": list(args.act_neg_band),
                },
                "rate_abs_max_kpa_s": [
                    args.ch_pos_rate_max,
                    args.ch_neg_rate_max,
                    args.act_pos_rate_max,
                    args.act_neg_rate_max,
                ],
                "high_frequency_cutoff_hz": args.hf_cutoff_hz,
                "actuator_hf_power_fraction_max": [
                    args.pos_hf_fraction_max,
                    args.neg_hf_fraction_max,
                ],
                "global_startup_excluded_sec": args.ignore_initial_sec,
            },
            "randomness": {
                "seeded": True,
                "families": list(FAMILY_ORDER),
                "independent_timing_per_valve": True,
                "full_domain_uniform_proposal_probability": args.full_domain_probability,
                "each_endpoint_proposal_probability": args.endpoint_probability,
                "local_proposal_sigma": args.local_sigma,
            },
            "optimisation": {
                "method": "per-block complete-prefix simulator replay candidate search",
                "candidates_per_block": args.candidates_per_block,
                "feedback_controller_used": False,
                "PID_used": False,
                "teacher_action_given_to_RL": False,
            },
            "validation": {
                "trajectory_metrics": metrics,
                "generator_float64_replay_rmse_kpa_full_pressure": replay_rmse,
                "generator_float64_replay_max_abs_kpa_full_pressure": replay_max_abs,
                "saved_float32_replay_rmse_kpa_full_pressure": stored_replay_rmse,
                "saved_float32_replay_max_abs_kpa_full_pressure": stored_replay_max_abs,
            },
            "schedule": [item for trajectory in schedules for item in trajectory],
            "note": (
                "pressure is a deterministic lib3 simulator output. physical_action/raw_action "
                "are oracle replay diagnostics and are not loaded into the policy observation."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)

    plot_reference(time, pressure_bank[0], action_bank[0], q_bank[0], args)
    print(f"[ INFO] profile: {args.output}")
    print(f"[ INFO] trajectory bank: {args.npz_output}")
    print(f"[ INFO] visualization: {args.plot_output}")
    print(
        "[ INFO] saved-float32 replay RMSE "
        f"[ch_pos,ch_neg,act_pos,act_neg]: {stored_replay_rmse}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=61098)
    parser.add_argument("--bank-size", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--candidates-per-block", type=int, default=16)
    parser.add_argument("--command-domain", type=parse_pair, default=parse_pair("0.8,1.0"))
    parser.add_argument("--initial-physical-command", type=float, default=0.95)
    parser.add_argument("--initial-hold-sec", type=float, default=0.6)
    parser.add_argument("--transition-count", type=lambda text: tuple(map(int, text.split(","))), default=(1, 3))
    parser.add_argument("--transition-sec", type=parse_pair, default=parse_pair("0.8,3.5"))
    parser.add_argument("--hold-sec", type=parse_pair, default=parse_pair("0.8,7.0"))
    parser.add_argument("--min-command-delta", type=float, default=0.012)
    parser.add_argument("--full-domain-probability", type=float, default=0.20)
    parser.add_argument("--endpoint-probability", type=float, default=0.04)
    parser.add_argument("--local-sigma", type=float, default=0.025)

    parser.add_argument("--ch-pos-band", type=parse_pair, default=parse_pair("95,220"))
    parser.add_argument("--ch-neg-band", type=parse_pair, default=parse_pair("5,105"))
    parser.add_argument("--act-pos-band", type=parse_pair, default=parse_pair("95,180"))
    parser.add_argument("--act-neg-band", type=parse_pair, default=parse_pair("25,105"))
    parser.add_argument("--ch-pos-rate-max", type=float, default=80.0)
    parser.add_argument("--ch-neg-rate-max", type=float, default=50.0)
    parser.add_argument("--act-pos-rate-max", type=float, default=45.0)
    parser.add_argument("--act-neg-rate-max", type=float, default=20.0)
    parser.add_argument("--min-pos-span", type=float, default=7.0)
    parser.add_argument("--min-neg-span", type=float, default=5.0)
    parser.add_argument("--ignore-initial-sec", type=float, default=1.0)
    parser.add_argument("--hf-cutoff-hz", type=float, default=2.0)
    parser.add_argument("--pos-hf-fraction-max", type=float, default=0.12)
    parser.add_argument("--neg-hf-fraction-max", type=float, default=0.18)
    parser.add_argument("--terminal-ch-pos", type=parse_pair, default=parse_pair("105,205"))
    parser.add_argument("--terminal-ch-neg", type=parse_pair, default=parse_pair("10,95"))
    parser.add_argument("--terminal-act-pos", type=parse_pair, default=parse_pair("105,170"))
    parser.add_argument("--terminal-act-neg", type=parse_pair, default=parse_pair("35,95"))

    parser.add_argument("--emergency-ch-pos", type=parse_pair, default=parse_pair("80,250"))
    parser.add_argument("--emergency-ch-neg", type=parse_pair, default=parse_pair("1,115"))
    parser.add_argument("--emergency-act-pos", type=parse_pair, default=parse_pair("80,220"))
    parser.add_argument("--emergency-act-neg", type=parse_pair, default=parse_pair("1,115"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pneu_rl/config/reference/lib3_full_domain_oracle_hard_4x512_seed61098.yaml"),
    )
    parser.add_argument(
        "--npz-output",
        type=Path,
        default=Path("pneu_rl/config/reference/lib3_full_domain_oracle_hard_4x512_seed61098.npz"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("pneu_rl/img/reference/lib3_full_domain_oracle_hard_4x512_seed61098.png"),
    )
    args = parser.parse_args()

    if args.bank_size < 1 or args.blocks < 1 or args.horizon_steps < 16:
        parser.error("bank-size, blocks, and horizon-steps must be positive")
    if args.candidates_per_block < 1:
        parser.error("candidates-per-block must be positive")
    if len(args.transition_count) != 2 or args.transition_count[0] > args.transition_count[1]:
        parser.error("--transition-count must be an ascending pair such as 1,3")
    if not (0.0 <= args.command_domain[0] < args.command_domain[1] <= 1.0):
        parser.error("--command-domain must lie inside [0,1]")
    if not args.command_domain[0] <= args.initial_physical_command <= args.command_domain[1]:
        parser.error("--initial-physical-command must lie inside --command-domain")
    if 2.0 * args.endpoint_probability + args.full_domain_probability > 1.0:
        parser.error("endpoint/full-domain proposal probabilities sum to more than one")
    args.emergency_bands = (
        args.emergency_ch_pos,
        args.emergency_ch_neg,
        args.emergency_act_pos,
        args.emergency_act_neg,
    )

    actions: list[np.ndarray] = []
    pressures: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    z_values: list[np.ndarray] = []
    mdot_values: list[np.ndarray] = []
    schedules: list[list[dict[str, Any]]] = []
    visited: set[tuple[int, int, int, int]] = set()
    for trajectory_index in range(args.bank_size):
        action, pressure, q_pred, valve_z, mdot, schedule = generate_one(
            args,
            trajectory_index=trajectory_index,
            visited=visited,
        )
        actions.append(action)
        pressures.append(pressure)
        q_values.append(q_pred)
        z_values.append(valve_z)
        mdot_values.append(mdot)
        schedules.append(schedule)
    save_bank(args, actions, pressures, q_values, z_values, mdot_values, schedules)


if __name__ == "__main__":
    main()
