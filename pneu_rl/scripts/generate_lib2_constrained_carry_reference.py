#!/usr/bin/env python3
"""Build a random, simulator-constrained lib2 carry reference.

This is deliberately *not* the legacy ``target -> feedback -> pressure``
generator.  It samples a task family and a continuous command programme for
each 512-step carry block, rolls every candidate through the real lib2 C++
simulator from the accumulated history, and commits the highest-scoring
constraint-satisfying programme.  The simulator pressure output is the RL
reference.  ``raw_action`` is saved only for oracle validation; SAC never
receives it.

The per-block candidate search is an offline derivative-free optimisation:
the task specification is random, while candidate selection maximises
pressure/rate coverage subject to pressure, rate, command and continuity
constraints.  No target tracking controller, PID, guard feedback or dither
is present in the generated trajectory.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source_dir in (ROOT / "pneu_env" / "src", ROOT / "pneu_ref" / "src", ROOT / "pneu_utils" / "src"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.sim2 import PneuSim


def parse_pair(text: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 2 or values[0] > values[1]:
        raise argparse.ArgumentTypeError("expected ascending comma-separated pair")
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
    """Allocate transition/hold durations that fill a block exactly."""
    minimum = count * (transition_range[0] + hold_range[0])
    capacities = [transition_range[1] - transition_range[0]] * count
    capacities += [hold_range[1] - hold_range[0]] * count
    if minimum > duration + 1e-9 or minimum + sum(capacities) < duration - 1e-9:
        raise ValueError("transition/hold ranges cannot fill one carry block")
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


def axis_program(
    time: np.ndarray,
    rng: random.Random,
    initial: float,
    command_range: tuple[float, float],
    count_range: tuple[int, int],
    transition_range: tuple[float, float],
    hold_range: tuple[float, float],
    min_delta: float,
    amplitude_scale: float,
    first_direction: str | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Sample a continuous piecewise-quintic command starting at ``initial``."""
    duration = float(time[-1])
    possible = [
        count for count in range(count_range[0], count_range[1] + 1)
        if count * (transition_range[0] + hold_range[0]) <= duration + 1e-9
        and count * (transition_range[1] + hold_range[1]) >= duration - 1e-9
    ]
    if not possible:
        raise ValueError("no valid transition count for carry block")
    count = rng.choice(possible)
    transitions, holds = allocate_durations(rng, count, duration, transition_range, hold_range)
    lower, upper = command_range
    values = np.full(len(time), float(initial), dtype=np.float64)
    current = float(initial)
    cursor = 0.0
    segments: list[dict[str, float]] = []
    max_jump = (upper - lower) * float(np.clip(amplitude_scale, 0.05, 1.0))
    for segment_index, (transition, hold) in enumerate(zip(transitions, holds)):
        local_lower = max(lower, current - max_jump)
        local_upper = min(upper, current + max_jump)
        # Force only the first segment when requested.  This prevents the
        # first task from being a long, uninformative drift away from the
        # agreed operating point; all later segments stay independent/random.
        if segment_index == 0 and first_direction == "lower":
            directed_upper = min(local_upper, current - min_delta)
            target = rng.uniform(local_lower, directed_upper) if directed_upper >= local_lower else rng.uniform(local_lower, local_upper)
        elif segment_index == 0 and first_direction == "upper":
            directed_lower = max(local_lower, current + min_delta)
            target = rng.uniform(directed_lower, local_upper) if directed_lower <= local_upper else rng.uniform(local_lower, local_upper)
        else:
            target = rng.uniform(local_lower, local_upper)
        for _ in range(16):
            if abs(target - current) >= min_delta:
                break
            target = rng.uniform(local_lower, local_upper)
        start = cursor
        end = start + transition
        mask = (time >= start) & (time < end)
        if np.any(mask):
            values[mask] = current + (target - current) * quintic((time[mask] - start) / transition)
        values[time >= end] = target
        cursor = end + hold
        segments.append(
            {
                "start_sec": float(start), "transition_sec": float(transition), "hold_sec": float(hold),
                "raw_start": float(current), "raw_target": float(target), "end_sec": float(cursor),
            }
        )
        current = target
    return values, segments


def axis_program_after_initial_hold(
    time: np.ndarray,
    rng: random.Random,
    initial: float,
    initial_hold_sec: float,
    command_range: tuple[float, float],
    count_range: tuple[int, int],
    transition_range: tuple[float, float],
    hold_range: tuple[float, float],
    min_delta: float,
    amplitude_scale: float,
    first_direction: str | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Keep the agreed initial operating point before the first task change."""
    if initial_hold_sec <= 0.0:
        return axis_program(
            time, rng, initial, command_range, count_range, transition_range, hold_range, min_delta, amplitude_scale,
            first_direction,
        )
    if initial_hold_sec >= float(time[-1]) - transition_range[0] - hold_range[0]:
        raise ValueError("initial hold leaves no room for a configured task transition")
    values = np.full(len(time), initial, dtype=np.float64)
    mask = time >= initial_hold_sec
    tail_time = time[mask] - initial_hold_sec
    tail, segments = axis_program(
        tail_time, rng, initial, command_range, count_range, transition_range, hold_range, min_delta, amplitude_scale,
        first_direction,
    )
    values[mask] = tail
    for segment in segments:
        segment["start_sec"] += initial_hold_sec
        segment["end_sec"] += initial_hold_sec
    return values, segments


FAMILIES: dict[str, dict[str, tuple[float, float] | tuple[int, int]]] = {
    # Command ranges are raw SAC actions.  POS direction is intentionally
    # inverted physically (smaller command closes the positive valve).
    "pos_sweep": {"pos_amp": (0.70, 1.00), "neg_amp": (0.10, 0.38), "pos_count": (2, 4), "neg_count": (1, 2)},
    "neg_sweep": {"pos_amp": (0.12, 0.38), "neg_amp": (0.72, 1.00), "pos_count": (1, 2), "neg_count": (2, 5)},
    "opposite": {"pos_amp": (0.50, 0.92), "neg_amp": (0.50, 0.92), "pos_count": (2, 4), "neg_count": (2, 5)},
    "mixed": {"pos_amp": (0.35, 0.82), "neg_amp": (0.35, 0.86), "pos_count": (1, 4), "neg_count": (1, 5)},
}


def simulate(actions: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Replay the complete committed prefix; preserves all hidden dynamics."""
    sim = PneuSim(
        freq=args.freq, delay=0.0, init_pos_press=args.reference_init_pos, init_neg_press=args.reference_init_neg,
        noise=False, scale=True, scale_low=args.scale_low, scale_high=args.scale_high,
    )
    sim.set_init_press(args.reference_init_pos, args.reference_init_neg)
    pressure = np.empty_like(actions, dtype=np.float64)
    pressure[0] = [args.reference_init_pos, args.reference_init_neg]
    for index in range(1, len(actions)):
        observation, _ = sim.observe(actions[index])
        pressure[index] = observation[1:3]
        # Candidate generation must never let an obviously unsafe transient
        # continue simply because it will be rejected only after a complete
        # 10.24-s rollout.  This is an *offline simulator guard*, not a
        # projection or restriction on SAC actions at training time.
        if (
            not np.all(np.isfinite(pressure[index]))
            or pressure[index, 0] < args.emergency_pos_min
            or pressure[index, 0] > args.emergency_pos_max
            or pressure[index, 1] < args.emergency_neg_min
            or pressure[index, 1] > args.emergency_neg_max
        ):
            pressure[index:] = np.nan
            break
    return pressure


def feature_bins(pressure: np.ndarray, rate: np.ndarray, args: argparse.Namespace) -> set[tuple[int, int, int, int]]:
    """Coarse pressure/rate bins used only in the offline coverage objective."""
    pos_norm = np.clip((pressure[:, 0] - args.pos_min) / (args.pos_max - args.pos_min), 0.0, 0.999)
    neg_norm = np.clip((pressure[:, 1] - args.neg_min) / (args.neg_max - args.neg_min), 0.0, 0.999)
    pos_rate = np.digitize(rate[:, 0], [-2.0, 0.0, 2.0])
    neg_rate = np.digitize(rate[:, 1], [-0.8, 0.0, 0.8])
    return {
        (int(pos * 6), int(neg * 5), int(pos_r), int(neg_r))
        for pos, neg, pos_r, neg_r in zip(pos_norm, neg_norm, pos_rate, neg_rate)
    }


def score_block(
    pressure: np.ndarray,
    actions: np.ndarray,
    start: int,
    visited: set[tuple[int, int, int, int]],
    family: str,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]] | None:
    dt = 1.0 / args.freq
    # The agreed reset/carry task starts with ref=(150, 24) while the learner
    # plant starts at atmosphere.  The reference-side simulator also has its
    # own initial pump transient.  Filter only this *single global startup*;
    # every later 512-step boundary remains fully constrained.
    checked_start = start + (int(round(args.ignore_initial_sec * args.freq)) if start == 0 else 0)
    block = pressure[checked_start:]
    rate = np.gradient(pressure, dt, axis=0)[checked_start:]
    if len(block) < 4 or not np.all(np.isfinite(block)):
        return None
    if (
        np.any(block[:, 0] < args.pos_min) or np.any(block[:, 0] > args.pos_max)
        or np.any(block[:, 1] < args.neg_min) or np.any(block[:, 1] > args.neg_max)
        or np.max(np.abs(rate[:, 0])) > args.pos_rate_max
        or np.max(np.abs(rate[:, 1])) > args.neg_rate_max
    ):
        return None
    span = np.ptp(block, axis=0)
    if span[0] < args.min_pos_span or span[1] < args.min_neg_span:
        return None
    if start == 0 and np.max(block[:, 0]) < args.reference_init_pos + args.initial_pos_rise_min:
        return None
    # A carried trajectory must leave enough room for the *next* random
    # block.  Without this terminal reserve, a locally feasible candidate can
    # end at the pressure boundary and make the following block infeasible.
    endpoint = block[-1]
    if (
        endpoint[0] < args.pos_min + args.pos_terminal_margin
        or endpoint[0] > args.pos_max - args.pos_terminal_margin
        or endpoint[1] < args.neg_min + args.neg_terminal_margin
        or endpoint[1] > args.neg_max - args.neg_terminal_margin
    ):
        return None
    bins = feature_bins(block, rate, args)
    gain = len(bins - visited)
    activity = np.mean(np.abs(rate), axis=0)
    # The requested family is only a diversity preference, never a pressure
    # target.  A candidate is selected by measured simulator output.
    family_bonus = 0.0
    if family == "pos_sweep":
        family_bonus = max(0.0, span[0] - 1.5 * span[1]) / max(args.pos_max - args.pos_min, 1.0)
    elif family == "neg_sweep":
        family_bonus = max(0.0, span[1] * 4.0 - span[0]) / max(args.neg_max - args.neg_min, 1.0)
    elif family == "opposite":
        family_bonus = min(activity[0] / args.pos_rate_max, activity[1] / args.neg_rate_max)
    score = (
        4.0 * gain
        + 1.5 * span[0] / (args.pos_max - args.pos_min)
        + 1.5 * span[1] / (args.neg_max - args.neg_min)
        + 0.25 * activity[0] / args.pos_rate_max
        + 0.25 * activity[1] / args.neg_rate_max
        + family_bonus
    )
    metadata = {
        "score": float(score), "coverage_gain": int(gain), "pressure_min": block.min(axis=0).tolist(),
        "pressure_max": block.max(axis=0).tolist(), "pressure_span": span.tolist(),
        "rate_abs_max": np.max(np.abs(rate), axis=0).tolist(), "rate_mean_abs": activity.tolist(),
        "raw_action_min": actions[start:].min(axis=0).tolist(), "raw_action_max": actions[start:].max(axis=0).tolist(),
    }
    return score, metadata


def rejection_reason(pressure: np.ndarray, start: int, args: argparse.Namespace) -> str:
    """Small diagnostic used when an offline candidate pool is empty."""
    checked_start = start + (int(round(args.ignore_initial_sec * args.freq)) if start == 0 else 0)
    block = pressure[checked_start:]
    rate = np.gradient(pressure, 1.0 / args.freq, axis=0)[checked_start:]
    if not np.all(np.isfinite(block)):
        return "nonfinite"
    if np.any(block[:, 0] < args.pos_min) or np.any(block[:, 0] > args.pos_max):
        return "pos_band"
    if np.any(block[:, 1] < args.neg_min) or np.any(block[:, 1] > args.neg_max):
        return "neg_band"
    if np.max(np.abs(rate[:, 0])) > args.pos_rate_max:
        return "pos_rate"
    if np.max(np.abs(rate[:, 1])) > args.neg_rate_max:
        return "neg_rate"
    span = np.ptp(block, axis=0)
    if span[0] < args.min_pos_span:
        return "pos_span"
    if span[1] < args.min_neg_span:
        return "neg_span"
    if start == 0 and np.max(block[:, 0]) < args.reference_init_pos + args.initial_pos_rise_min:
        return "initial_pos_rise"
    endpoint = block[-1]
    if endpoint[0] < args.pos_min + args.pos_terminal_margin or endpoint[0] > args.pos_max - args.pos_terminal_margin:
        return "pos_terminal_margin"
    if endpoint[1] < args.neg_min + args.neg_terminal_margin or endpoint[1] > args.neg_max - args.neg_terminal_margin:
        return "neg_terminal_margin"
    return "unknown"


def propose_block(
    rng: random.Random,
    previous_action: np.ndarray,
    family: str,
    args: argparse.Namespace,
    initial_hold_sec: float = 0.0,
    initial_task: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Include the shared boundary sample.  The caller drops this first sample
    # when appending, so each 512-step block contributes exactly 512 physical
    # steps (= 10.24 s at 50 Hz) without a command jump at the boundary.
    block_time = np.arange(args.horizon_steps + 1, dtype=np.float64) / args.freq
    setting = FAMILIES[family]
    pos_amp = rng.uniform(*setting["pos_amp"])  # type: ignore[arg-type]
    neg_amp = rng.uniform(*setting["neg_amp"])  # type: ignore[arg-type]
    # In lib2 a lower POS command closes the positive valve and raises P_pos.
    # Start with a real excursion after the initial hold.  NEG's first
    # direction is selected independently so the two axes are not locked.
    pos_first_direction = "lower" if initial_task else None
    neg_first_direction = rng.choice(["lower", "upper"]) if initial_task else None
    pos, pos_segments = axis_program_after_initial_hold(
        block_time, rng, float(previous_action[0]), initial_hold_sec, args.raw_pos, setting["pos_count"],
        args.transition_sec, args.hold_sec, args.min_raw_delta, pos_amp, pos_first_direction,  # type: ignore[arg-type]
    )
    neg, neg_segments = axis_program_after_initial_hold(
        block_time, rng, float(previous_action[1]), initial_hold_sec, args.raw_neg, setting["neg_count"],
        args.transition_sec, args.hold_sec, args.min_raw_delta, neg_amp, neg_first_direction,  # type: ignore[arg-type]
    )
    return np.column_stack((pos, neg)), {
        "family": family, "pos_amplitude_scale": float(pos_amp), "neg_amplitude_scale": float(neg_amp),
        "pos_segments": pos_segments, "neg_segments": neg_segments,
        "pos_first_direction_raw": pos_first_direction, "neg_first_direction_raw": neg_first_direction,
    }


def generate(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    committed = np.asarray([[args.initial_raw_pos, args.initial_raw_neg]], dtype=np.float64)
    pressure = np.asarray([[args.reference_init_pos, args.reference_init_neg]], dtype=np.float64)
    visited: set[tuple[int, int, int, int]] = set()
    schedule: list[dict[str, Any]] = []
    families = list(FAMILIES)
    for block_index in range(args.episodes):
        # Random task specification.  Cycling the candidate order prevents one
        # attractive family from starving the NEG-only or coupled cases.
        # A real initial POS excursion after the 150/24 operating-point hold;
        # later blocks cycle through all task families for balanced coverage.
        family = "mixed" if block_index == 0 else families[block_index % len(families)]
        best: tuple[float, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]] | None = None
        rejected: dict[str, int] = {}
        for candidate_index in range(args.candidates_per_block):
            proposal, proposal_meta = propose_block(
                rng, committed[-1], family, args,
                initial_hold_sec=args.initial_hold_sec if block_index == 0 else 0.0,
                initial_task=block_index == 0,
            )
            candidate_actions = np.vstack((committed, proposal[1:]))
            candidate_pressure = simulate(candidate_actions, args)
            result = score_block(candidate_pressure, candidate_actions, len(committed) - 1, visited, family, args)
            if result is None:
                reason = rejection_reason(candidate_pressure, len(committed) - 1, args)
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            score, score_meta = result
            if best is None or score > best[0]:
                best = (score, candidate_actions, candidate_pressure, proposal_meta, score_meta)
        if best is None:
            raise RuntimeError(
                f"no feasible candidate at carry block {block_index}; rejection counts: {rejected}"
            )
        _, committed, pressure, proposal_meta, score_meta = best
        block_start = len(committed) - args.horizon_steps
        block_rate = np.gradient(pressure, 1.0 / args.freq, axis=0)[block_start:]
        visited.update(feature_bins(pressure[block_start:], block_rate, args))
        schedule.append({
            "block": block_index, "start_sec": float(block_index * args.horizon_steps / args.freq),
            "family": family, "candidate_count": args.candidates_per_block, **proposal_meta, **score_meta,
        })
        if (block_index + 1) % args.progress_every_blocks == 0 or block_index + 1 == args.episodes:
            print(
                f"[ INFO] carry-opt | block {block_index + 1}/{args.episodes} | family {family}"
                f" | P {pressure[-1, 0]:.2f},{pressure[-1, 1]:.2f} | coverage {len(visited)}"
            )
    return committed, pressure, schedule


def save(args: argparse.Namespace, actions: np.ndarray, pressure: np.ndarray, schedule: list[dict[str, Any]]) -> None:
    dt = 1.0 / args.freq
    time = np.arange(len(actions), dtype=np.float64) * dt
    rate = np.gradient(pressure, dt, axis=0)
    start = min(len(pressure) - 1, int(round(args.ignore_initial_sec * args.freq)))
    checked_pressure = pressure[start:]
    checked_rate = rate[start:]
    metadata = {
        "samples": int(len(actions)), "duration_sec": float(time[-1]), "episode_boundaries_512": args.episodes,
        "pressure_min_after_startup": checked_pressure.min(axis=0).tolist(),
        "pressure_max_after_startup": checked_pressure.max(axis=0).tolist(),
        "pressure_span_after_startup": np.ptp(checked_pressure, axis=0).tolist(),
        "reference_rate_abs_max_after_startup": np.max(np.abs(checked_rate), axis=0).tolist(),
        "action_min_raw": actions.min(axis=0).tolist(), "action_max_raw": actions.max(axis=0).tolist(),
    }
    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz_output, time=time.astype(np.float32), pressure=pressure.astype(np.float32), raw_action=actions.astype(np.float32))
    profile = {
        "kind": "optimized_trajectory_bank",
        "kwargs": {"trajectory_npz": str(args.npz_output), "seed": args.seed, "sample_mode": "coverage", "repeat_after_end": False},
        "design": {
            "library": 2, "continuous_carry": True, "duration_sec": metadata["duration_sec"],
            "horizon_512_sec": args.horizon_steps / args.freq, "episode_boundaries_inside_rollout": args.episodes,
            "generator": "offline constrained random task specification with simulator-in-the-loop command-program optimisation",
            "reference_initial_pressure_kpa": [args.reference_init_pos, args.reference_init_neg],
            "expected_training_initial_pressure_kpa": [101.325, 101.325],
            "rate_filter_ignore_initial_sec": args.ignore_initial_sec,
            "pressure_filter_kpa": {"pos": list(args.pos_band), "neg": list(args.neg_band)},
            "rate_filter_kpa_s": [args.pos_rate_max, args.neg_rate_max],
            "raw_command_range": {"pos": list(args.raw_pos), "neg": list(args.raw_neg)},
            "randomness": {
                "seeded": True, "task_families": list(FAMILIES), "independent_pos_neg_timing": True,
                "random_variables": ["family", "transition_count", "transition_duration", "hold_duration", "command_target", "amplitude"],
            },
            "optimisation": {
                "method": "per-block simulator replay candidate search", "candidates_per_block": args.candidates_per_block,
                "objective": "pressure/rate coverage gain + span + activity, subject to all hard constraints",
                "feedback_controller_used": False, "dither_used": False,
            },
            "schedule": schedule, "validation": metadata,
            "note": "The stored pressure is a simulator output of selected continuous command programmes. raw_action is oracle-only and is not given to RL.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(time, pressure[:, 0], color="red", label="POS ref")
    axes[1].plot(time, pressure[:, 1], color="blue", label="NEG ref")
    axes[2].plot(time, rate[:, 0], color="darkred", label="dPOS ref/dt")
    axes[2].plot(time, rate[:, 1], color="navy", label="dNEG ref/dt")
    axes[3].plot(time, actions[:, 0], color="red", label="POS oracle raw action")
    axes[3].plot(time, actions[:, 1], color="blue", label="NEG oracle raw action")
    for boundary in range(1, args.episodes):
        for axis in axes:
            axis.axvline(boundary * args.horizon_steps / args.freq, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("POS ref [kPa]")
    axes[1].set_ylabel("NEG ref [kPa]")
    axes[2].set_ylabel("dP/dt [kPa/s]")
    axes[3].set_ylabel("raw action")
    axes[3].set_xlabel("Continuous time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("lib2 constrained random carry reference (no target feedback / no dither)")
    fig.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] profile: {args.output}")
    print(f"[ INFO] metadata: {metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=743)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon-steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--candidates-per-block", type=int, default=6)
    parser.add_argument("--progress-every-blocks", type=int, default=1)
    parser.add_argument("--reference-init-pos", type=float, default=150.0)
    parser.add_argument("--reference-init-neg", type=float, default=24.0)
    parser.add_argument("--initial-raw-pos", type=float, default=0.50)
    parser.add_argument("--initial-raw-neg", type=float, default=0.87)
    parser.add_argument("--initial-hold-sec", type=float, default=1.0)
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--scale-high", type=float, default=1.0)
    # Normalized SAC action interval.  The simulator maps this to
    # [scale_low, scale_high]; do not silently remove the strong closing
    # region from reference design while policy training can use it.
    parser.add_argument("--raw-pos", type=parse_pair, default=parse_pair("0.00,0.90"))
    parser.add_argument("--raw-neg", type=parse_pair, default=parse_pair("0.15,1.00"))
    parser.add_argument("--transition-sec", type=parse_pair, default=parse_pair("0.45,2.20"))
    parser.add_argument("--hold-sec", type=parse_pair, default=parse_pair("0.80,6.0"))
    parser.add_argument("--min-raw-delta", type=float, default=0.055)
    parser.add_argument("--pos-band", type=parse_pair, default=parse_pair("115,220"))
    parser.add_argument("--neg-band", type=parse_pair, default=parse_pair("10,30"))
    parser.add_argument("--emergency-pos-min", type=float, default=90.0)
    parser.add_argument("--emergency-pos-max", type=float, default=250.0)
    parser.add_argument("--emergency-neg-min", type=float, default=5.0)
    parser.add_argument("--emergency-neg-max", type=float, default=35.0)
    parser.add_argument("--pos-rate-max", type=float, default=32.0)
    parser.add_argument("--neg-rate-max", type=float, default=9.5)
    parser.add_argument("--min-pos-span", type=float, default=10.0)
    parser.add_argument("--min-neg-span", type=float, default=1.5)
    parser.add_argument("--initial-pos-rise-min", type=float, default=8.0)
    parser.add_argument("--pos-terminal-margin", type=float, default=8.0)
    parser.add_argument("--neg-terminal-margin", type=float, default=1.5)
    parser.add_argument("--ignore-initial-sec", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    args.pos_min, args.pos_max = args.pos_band
    args.neg_min, args.neg_max = args.neg_band
    if args.episodes < 1 or args.horizon_steps < 4 or args.candidates_per_block < 1:
        parser.error("episodes, horizon-steps and candidates-per-block must be positive")
    if not (0.0 <= args.raw_pos[0] <= args.raw_pos[1] <= 1.0 and 0.0 <= args.raw_neg[0] <= args.raw_neg[1] <= 1.0):
        parser.error("raw command ranges must be within [0, 1]")
    actions, pressure, schedule = generate(args)
    save(args, actions, pressure, schedule)


if __name__ == "__main__":
    main()
