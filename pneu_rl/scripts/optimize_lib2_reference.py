#!/usr/bin/env python3
"""Build a simulator-backed, randomized reference trajectory bank for lib2.

The optimizer is deliberately lightweight and reproducible.  It samples
independent continuous valve-command waveforms, rolls each waveform through
the complete lib2 simulator, rejects pressure/rate-infeasible rollouts, and
keeps a diverse subset.  The resulting pressure trajectories are therefore
jointly feasible by construction; pressure min/max boxes are only used as
additional safety filters.

This is an offline reference-design tool, not an online optimizer inside SAC.
"""

from __future__ import annotations

import argparse
import csv
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
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_utils" / "src",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.sim2 import PneuSim


def parse_pair(text: str) -> tuple[float, float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated values")
    return float(values[0]), float(values[1])


def parse_int_pair(text: str) -> tuple[int, int]:
    values = [int(part.strip()) for part in text.split(",")]
    if len(values) != 2 or values[0] < 1 or values[1] < values[0]:
        raise argparse.ArgumentTypeError("expected ascending positive integer pair, e.g. 4,8")
    return values[0], values[1]


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / float(width)
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def count_turns(values: np.ndarray, min_distance: int = 10) -> int:
    """Count robust local extrema, ignoring one-step simulator ripple."""
    smooth_width = max(3, min_distance // 2 * 2 + 1)
    smooth = moving_average(np.asarray(values, dtype=np.float64), smooth_width)
    derivative = np.diff(smooth)
    signs = np.sign(derivative)
    signs[signs == 0.0] = 1.0
    last_sign = float(signs[0]) if len(signs) else 1.0
    last_index = 0
    turns = 0
    for index in range(1, len(signs)):
        sign = float(signs[index])
        if sign == last_sign:
            continue
        if index - last_index >= min_distance:
            turns += 1
            last_index = index
            last_sign = sign
    return int(turns)


def command_waveform(
    times: np.ndarray,
    base: np.ndarray,
    amplitude: np.ndarray,
    cycles: np.ndarray,
    phase: np.ndarray,
    harmonic_mix: np.ndarray,
    duration: float,
) -> np.ndarray:
    tau = times / max(duration, 1e-9)
    values = np.empty((len(times), 2), dtype=np.float64)
    for axis in range(2):
        fundamental = np.sin(2.0 * math.pi * cycles[axis] * tau + phase[axis])
        second = np.sin(2.0 * math.pi * (cycles[axis] + 0.5) * tau + 1.7 * phase[axis])
        signal = (1.0 - harmonic_mix[axis]) * fundamental + harmonic_mix[axis] * second
        values[:, axis] = base[axis] + amplitude[axis] * signal
    return np.clip(values, 0.0, 1.0)


def quintic_progress(progress: np.ndarray) -> np.ndarray:
    """Minimum-jerk progress: value, velocity and acceleration are zero at both ends."""
    tau = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def segmented_command_waveform(
    times: np.ndarray,
    rng: random.Random,
    raw_range: tuple[float, float],
    segment_count_range: tuple[int, int],
    transition_range: tuple[float, float],
    hold_range: tuple[float, float],
    min_target_delta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create one axis with explicit, random transition and hold durations.

    The endpoint commands, transition lengths, and hold lengths are sampled
    independently for each pressure axis.  A quintic segment gives a continuous
    command with zero endpoint slope; the complete command is later rolled out
    in lib2, so pressure feasibility is still decided by the simulator.
    """
    duration = float(times[-1])
    raw_lower, raw_upper = raw_range
    min_transition, max_transition = transition_range
    min_hold, max_hold = hold_range
    feasible_counts = [
        count
        for count in range(segment_count_range[0], segment_count_range[1] + 1)
        if count * (min_transition + min_hold) <= duration + 1e-9
        and count * (max_transition + max_hold) >= duration - 1e-9
    ]
    if not feasible_counts:
        raise ValueError(
            "transition/hold ranges cannot fill the episode horizon; increase --hold-sec upper bound, "
            "increase segment count, or reduce duration"
        )
    requested_count = rng.choice(feasible_counts)
    values = np.empty(len(times), dtype=np.float64)
    current = rng.uniform(raw_lower, raw_upper)
    values.fill(current)
    # Allocate exactly the whole horizon while keeping *every* transition and
    # hold within the declared ranges.  This fixes the common failure mode in
    # which the final waypoint silently becomes a very long unconfigured hold.
    duration_min = requested_count * (min_transition + min_hold)
    extra = duration - duration_min
    capacities = [max_transition - min_transition] * requested_count + [max_hold - min_hold] * requested_count
    increments = [0.0] * len(capacities)
    remaining = extra
    remaining_capacity = float(sum(capacities))
    indices = list(range(len(capacities)))
    rng.shuffle(indices)
    for order, index in enumerate(indices):
        capacity = capacities[index]
        remaining_capacity -= capacity
        allocation_lower = max(0.0, remaining - remaining_capacity)
        allocation_upper = min(capacity, remaining)
        increment = allocation_upper if order == len(indices) - 1 else rng.uniform(allocation_lower, allocation_upper)
        increments[index] = increment
        remaining -= increment
    transitions = [min_transition + increments[index] for index in range(requested_count)]
    holds = [min_hold + increments[requested_count + index] for index in range(requested_count)]

    cursor = 0.0
    segments: list[dict[str, float]] = []

    for segment_index, (transition, hold) in enumerate(zip(transitions, holds)):
        target = rng.uniform(raw_lower, raw_upper)
        for _ in range(12):
            if abs(target - current) >= min_target_delta:
                break
            target = rng.uniform(raw_lower, raw_upper)
        start_time = cursor
        end_time = cursor + transition
        transition_mask = (times >= start_time) & (times < end_time)
        if np.any(transition_mask):
            normalized = (times[transition_mask] - start_time) / max(transition, 1e-9)
            values[transition_mask] = current + (target - current) * quintic_progress(normalized)
        values[times >= end_time] = target
        hold_end = end_time + hold
        segments.append(
            {
                "start_sec": float(start_time),
                "transition_sec": float(transition),
                "hold_sec": float(hold),
                "transition_end_sec": float(end_time),
                "hold_end_sec": float(hold_end),
                "raw_start": float(current),
                "raw_target": float(target),
            }
        )
        current = target
        cursor = hold_end
    metadata = {
        "requested_transition_count": int(requested_count),
        "realized_transition_count": int(len(segments)),
        "transition_duration_sec": [float(item["transition_sec"]) for item in segments],
        "hold_duration_sec": [float(item["hold_sec"]) for item in segments],
        "segments": segments,
    }
    return np.clip(values, 0.0, 1.0), metadata


def simulate_candidate(args: argparse.Namespace, rng: random.Random, candidate_id: int) -> dict[str, Any] | None:
    duration = float(args.duration)
    dt = 1.0 / float(args.freq)
    times = np.arange(0.0, duration + 0.5 * dt, dt, dtype=np.float64)
    waveform_metadata: dict[str, Any]
    if args.waveform == "segmented":
        pos_action, pos_schedule = segmented_command_waveform(
            times,
            rng,
            args.raw_pos,
            args.segments_pos,
            args.transition_sec,
            args.hold_sec,
            args.min_target_delta,
        )
        neg_action, neg_schedule = segmented_command_waveform(
            times,
            rng,
            args.raw_neg,
            args.segments_neg,
            args.transition_sec,
            args.hold_sec,
            args.min_target_delta,
        )
        actions = np.column_stack((pos_action, neg_action))
        waveform_metadata = {
            "waveform": "independent_quintic_transition_hold",
            "pos_schedule": pos_schedule,
            "neg_schedule": neg_schedule,
        }
        # Retain common metadata keys so downstream tools can compare the
        # legacy sinusoid bank and this explicit transition/hold bank.
        base = np.asarray([pos_action[0], neg_action[0]], dtype=np.float64)
        amplitude = np.asarray([0.5 * np.ptp(pos_action), 0.5 * np.ptp(neg_action)], dtype=np.float64)
        cycles = np.asarray([float("nan"), float("nan")], dtype=np.float64)
        phase = np.asarray([float("nan"), float("nan")], dtype=np.float64)
        harmonic_mix = np.asarray([0.0, 0.0], dtype=np.float64)
    else:
        base = np.asarray(
            [rng.uniform(args.base_pos[0], args.base_pos[1]), rng.uniform(args.base_neg[0], args.base_neg[1])],
            dtype=np.float64,
        )
        amplitude = np.asarray(
            [rng.uniform(args.amp_pos[0], args.amp_pos[1]), rng.uniform(args.amp_neg[0], args.amp_neg[1])],
            dtype=np.float64,
        )
        cycles = np.asarray(
            [rng.uniform(args.cycles_pos[0], args.cycles_pos[1]), rng.uniform(args.cycles_neg[0], args.cycles_neg[1])],
            dtype=np.float64,
        )
        phase = np.asarray([rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)], dtype=np.float64)
        harmonic_mix = np.asarray([rng.uniform(0.0, 0.35), rng.uniform(0.0, 0.35)], dtype=np.float64)
        actions = command_waveform(times, base, amplitude, cycles, phase, harmonic_mix, duration)
        waveform_metadata = {"waveform": "independent_sinusoidal_harmonic"}

    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    sim.set_debug_enabled(True)
    sim.set_init_press(args.init_pos, args.init_neg)
    pressure = np.empty((len(times), 2), dtype=np.float64)
    pressure[0] = [args.init_pos, args.init_neg]
    model_rates = np.zeros((len(times), 2), dtype=np.float64)
    for index in range(1, len(times)):
        obs, _ = sim.observe(actions[index], goal=pressure[index - 1])
        pressure[index] = obs[1:3]
        debug = sim.get_model_debug()
        model_rates[index] = [debug["dPposdt"], debug["dPnegdt"]]
    del sim

    check_start = int(round(args.ignore_initial_sec * args.freq))
    checked_pressure = pressure[min(check_start, len(pressure) - 1) :]
    reference_rate = np.gradient(pressure, dt, axis=0)
    checked_rates = reference_rate[min(check_start, len(reference_rate) - 1) :]
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(model_rates)):
        return None
    if np.any(checked_pressure[:, 0] < args.pressure_pos_min) or np.any(checked_pressure[:, 0] > args.pressure_pos_max):
        return None
    if np.any(checked_pressure[:, 1] < args.pressure_neg_min) or np.any(checked_pressure[:, 1] > args.pressure_neg_max):
        return None
    if np.max(np.abs(checked_rates[:, 0])) > args.rate_pos_max or np.max(np.abs(checked_rates[:, 1])) > args.rate_neg_max:
        return None

    pressure_rate = reference_rate
    pos_span = float(np.ptp(checked_pressure[:, 0]))
    neg_span = float(np.ptp(checked_pressure[:, 1]))
    if pos_span < args.min_pos_span or neg_span < args.min_neg_span:
        return None
    pos_activity = float(np.mean(np.abs(pressure_rate[check_start:, 0])))
    neg_activity = float(np.mean(np.abs(pressure_rate[check_start:, 1])))
    score = (
        pos_span / max(args.pressure_pos_max - args.pressure_pos_min, 1e-9)
        + neg_span / max(args.pressure_neg_max - args.pressure_neg_min, 1e-9)
        + 0.15 * pos_activity / max(args.rate_pos_max, 1e-9)
        + 0.15 * neg_activity / max(args.rate_neg_max, 1e-9)
    )
    metadata = {
        "candidate_id": int(candidate_id),
        "score": float(score),
        "base_raw": base.tolist(),
        "amplitude_raw": amplitude.tolist(),
        "requested_cycles": cycles.tolist(),
        "phase_rad": phase.tolist(),
        "harmonic_mix": harmonic_mix.tolist(),
        "command_cycles": [float(cycles[0]), float(cycles[1])],
        "pressure_turns": [
            count_turns(checked_pressure[:, 0], max(10, int(args.freq * 0.35))),
            count_turns(checked_pressure[:, 1], max(10, int(args.freq * 0.35))),
        ],
        "pressure_min": checked_pressure.min(axis=0).tolist(),
        "pressure_max": checked_pressure.max(axis=0).tolist(),
        "pressure_span": [pos_span, neg_span],
        "pressure_rate_abs_max": np.max(np.abs(checked_rates), axis=0).tolist(),
        "simulator_model_rate_abs_max": np.max(np.abs(model_rates[check_start:]), axis=0).tolist(),
        "reference_rate_abs_max": np.max(np.abs(pressure_rate[check_start:]), axis=0).tolist(),
        "action_raw_min": actions.min(axis=0).tolist(),
        "action_raw_max": actions.max(axis=0).tolist(),
        "physical_action_min": (args.scale_low + (args.scale_high - args.scale_low) * actions.min(axis=0)).tolist(),
        "physical_action_max": (args.scale_low + (args.scale_high - args.scale_low) * actions.max(axis=0)).tolist(),
        **waveform_metadata,
    }
    return {
        "id": int(candidate_id),
        "time": times,
        "pressure": pressure,
        "raw_action": actions,
        "metadata": metadata,
        "feature": np.asarray(
            [
                checked_pressure[:, 0].min(),
                checked_pressure[:, 0].max(),
                checked_pressure[:, 1].min(),
                checked_pressure[:, 1].max(),
                pos_span,
                neg_span,
                metadata["pressure_turns"][0],
                metadata["pressure_turns"][1],
                float(metadata.get("pos_schedule", {}).get("realized_transition_count", 0)),
                float(metadata.get("neg_schedule", {}).get("realized_transition_count", 0)),
            ],
            dtype=np.float64,
        ),
    }


def select_diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(candidates) <= count:
        return sorted(candidates, key=lambda item: item["metadata"]["score"], reverse=True)
    ordered = sorted(candidates, key=lambda item: item["metadata"]["score"], reverse=True)
    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining and len(selected) < count:
        matrix = np.asarray([item["feature"] for item in selected], dtype=np.float64)
        scales = np.ptp(np.asarray([item["feature"] for item in candidates], dtype=np.float64), axis=0)
        scales[scales < 1e-9] = 1.0
        best_index = 0
        best_distance = -np.inf
        for index, item in enumerate(remaining):
            distance = np.min(np.linalg.norm((matrix - item["feature"]) / scales, axis=1))
            value = float(distance + 0.05 * item["metadata"]["score"])
            if value > best_distance:
                best_distance = value
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_bank(args: argparse.Namespace, selected: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_yaml = []
    summary_rows = []
    for item in selected:
        metadata = dict(item["metadata"])
        trajectory_yaml.append(
            {
                "id": int(item["id"]),
                "time": [float(value) for value in item["time"]],
                "pressure": [[float(value) for value in row] for row in item["pressure"]],
                "raw_action": [[float(value) for value in row] for row in item["raw_action"]],
                "metadata": metadata,
            }
        )
        summary_rows.append({"trajectory_id": int(item["id"]), **metadata})
    profile = {
        "kind": "optimized_trajectory_bank",
        "kwargs": {
            "trajectories": trajectory_yaml,
            "seed": int(args.seed),
            "sample_mode": "coverage",
            # A reset bank is intentionally not a carry bank.  Clamping after
            # its stored horizon is safer than silently jumping to a different
            # trajectory (or to an atmospheric startup point) at a carry
            # boundary.  Use generate_lib2_carry_reference.py for carry.
            "repeat_after_end": False,
        },
        "design": {
            "library": 2,
            "generator": "simulator-backed independent command transition/hold search",
            "waveform": args.waveform,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "duration_sec": args.duration,
            "frequency_hz": args.freq,
            "simulator_initial_pressure_kpa": [args.init_pos, args.init_neg],
            "physical_action_range": [args.scale_low, args.scale_high],
            "pressure_filter_kpa": {
                "pos": [args.pressure_pos_min, args.pressure_pos_max],
                "neg": [args.pressure_neg_min, args.pressure_neg_max],
            },
            "minimum_pressure_span_kpa": [args.min_pos_span, args.min_neg_span],
            "rate_filter_kpa_s": [args.rate_pos_max, args.rate_neg_max],
            "rate_filter_ignore_initial_sec": args.ignore_initial_sec,
            "timing_design": {
                "pos_transition_count_range": list(args.segments_pos),
                "neg_transition_count_range": list(args.segments_neg),
                "transition_sec_range": list(args.transition_sec),
                "hold_sec_range": list(args.hold_sec),
                "minimum_raw_target_delta": args.min_target_delta,
                "independent_axes": True,
            },
            "note": "All stored pressure pairs came from the complete lib2 simulator; filters are safety/diversity gates. POS/NEG schedules have independently sampled target values, transition durations, hold durations, and boundary times.",
        },
    }
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)
    write_csv(args.summary_csv, summary_rows)

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    colors = plt.cm.turbo(np.linspace(0.03, 0.97, len(selected)))
    for color, item in zip(colors, selected):
        label = f"traj {item['id']}"
        time = item["time"]
        pressure = item["pressure"]
        action = item["raw_action"]
        rate = np.gradient(pressure, 1.0 / args.freq, axis=0)
        axes[0].plot(time, pressure[:, 0], color=color, alpha=0.8, label=label)
        axes[1].plot(time, pressure[:, 1], color=color, alpha=0.8, label=label)
        axes[2].plot(time, rate[:, 0], color=color, alpha=0.7)
        axes[2].plot(time, rate[:, 1], color=color, alpha=0.35, linestyle="--")
        axes[3].plot(time, action[:, 0], color=color, alpha=0.7)
        axes[3].plot(time, action[:, 1], color=color, alpha=0.35, linestyle="--")
    axes[0].set_ylabel("POS ref [kPa]")
    axes[1].set_ylabel("NEG ref [kPa]")
    axes[2].set_ylabel("dP/dt [kPa/s]\nsolid POS / dashed NEG")
    axes[3].set_ylabel("raw action\nsolid POS / dashed NEG")
    axes[3].set_xlabel("Time [s]")
    axes[0].set_title("Simulator-backed optimized reference bank")
    axes[0].legend(ncol=5, fontsize=8, loc="upper right")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--candidates", type=int, default=80)
    parser.add_argument("--bank-size", type=int, default=10)
    parser.add_argument("--duration", type=float, default=40.96)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument(
        "--init-pos",
        type=float,
        default=150.0,
        help="reference-rollout initial POS pressure; training may still reset the plant at atmosphere",
    )
    parser.add_argument(
        "--init-neg",
        type=float,
        default=24.0,
        help="reference-rollout initial NEG pressure; training may still reset the plant at atmosphere",
    )
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--scale-high", type=float, default=1.0)
    parser.add_argument(
        "--waveform",
        choices=("segmented", "sinusoidal"),
        default="segmented",
        help="segmented uses independent quintic transition/hold schedules; sinusoidal preserves the legacy generator",
    )
    parser.add_argument("--raw-pos", type=parse_pair, default=parse_pair("0.20,0.80"))
    parser.add_argument("--raw-neg", type=parse_pair, default=parse_pair("0.63,0.94"))
    parser.add_argument("--segments-pos", type=parse_int_pair, default=parse_int_pair("2,5"))
    parser.add_argument("--segments-neg", type=parse_int_pair, default=parse_int_pair("3,6"))
    parser.add_argument("--transition-sec", type=parse_pair, default=parse_pair("0.45,2.20"))
    parser.add_argument(
        "--hold-sec",
        type=parse_pair,
        default=parse_pair("1.00,25.00"),
        help="per-transition hold range; every realized hold is constrained to this interval",
    )
    parser.add_argument(
        "--min-target-delta",
        type=float,
        default=0.06,
        help="minimum raw-command jump between two scheduled targets",
    )
    # Legacy sinusoidal controls are retained only to make old banks
    # reproducible; new target-bank generation uses --waveform segmented.
    parser.add_argument("--base-pos", type=parse_pair, default=parse_pair("0.45,0.55"))
    parser.add_argument("--base-neg", type=parse_pair, default=parse_pair("0.75,0.82"))
    parser.add_argument("--amp-pos", type=parse_pair, default=parse_pair("0.10,0.25"))
    parser.add_argument("--amp-neg", type=parse_pair, default=parse_pair("0.05,0.12"))
    parser.add_argument("--cycles-pos", type=parse_pair, default=parse_pair("1.0,5.0"))
    parser.add_argument("--cycles-neg", type=parse_pair, default=parse_pair("1.0,7.0"))
    parser.add_argument("--pressure-pos-min", type=float, default=115.0)
    parser.add_argument("--pressure-pos-max", type=float, default=220.0)
    parser.add_argument("--pressure-neg-min", type=float, default=10.0)
    parser.add_argument("--pressure-neg-max", type=float, default=30.0)
    parser.add_argument("--min-pos-span", type=float, default=35.0, help="reject a trajectory unless its post-startup POS span reaches this value")
    parser.add_argument("--min-neg-span", type=float, default=4.0, help="reject a trajectory unless its post-startup NEG span reaches this value")
    parser.add_argument("--rate-pos-max", type=float, default=32.0)
    parser.add_argument("--rate-neg-max", type=float, default=9.5)
    parser.add_argument(
        "--ignore-initial-sec",
        type=float,
        default=1.0,
        help="exclude the reference-rollout pump-state settling transient from filters; it remains in the stored trajectory",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/trajectory_summary.csv")
    parser.add_argument("--plot-output", type=Path, default=ROOT / "pneu_rl/img/reference/lib2_optimized_target_hold_bank_v2.png")
    args = parser.parse_args()
    if args.bank_size < 1 or args.candidates < args.bank_size:
        parser.error("require candidates >= bank-size >= 1")
    if args.transition_sec[0] <= 0.0 or args.hold_sec[0] <= 0.0:
        parser.error("transition and hold durations must be positive")

    rng = random.Random(args.seed)
    candidates: list[dict[str, Any]] = []
    for candidate_id in range(args.candidates):
        result = simulate_candidate(args, rng, candidate_id)
        if result is not None:
            candidates.append(result)
    if len(candidates) < args.bank_size:
        raise RuntimeError(f"only {len(candidates)} feasible candidates; reduce constraints or increase --candidates")
    selected = select_diverse(candidates, args.bank_size)
    save_bank(args, selected, candidates)
    print(f"[ INFO] feasible candidates: {len(candidates)}/{args.candidates}")
    print(f"[ INFO] selected bank: {args.output}")
    print(f"[ INFO] summary: {args.summary_csv}")
    print(f"[ INFO] preview: {args.plot_output}")
    for item in selected:
        print(f"[ INFO] trajectory {item['id']}: {item['metadata']}")


if __name__ == "__main__":
    main()
