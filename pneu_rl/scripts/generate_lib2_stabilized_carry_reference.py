#!/usr/bin/env python3
"""Generate a long, simulator-backed carry reference with bounded drift.

The stabilizing term is used only while generating the reference.  It keeps a
long open-loop command waveform from slowly leaving the pressure operating
band; the stored pressure trajectory is still produced by one continuous
lib2 simulator rollout and is later used as a fixed reference for RL.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys

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
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 2 or values[1] < values[0]:
        raise argparse.ArgumentTypeError("expected ascending comma-separated pair")
    return values[0], values[1]


def quintic_progress(progress: np.ndarray) -> np.ndarray:
    tau = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def independent_segmented_target(
    time: np.ndarray,
    rng: random.Random,
    value_range: tuple[float, float],
    transitions_per_horizon: tuple[float, float],
    horizon_sec: float,
    transition_sec: tuple[float, float],
    hold_sec: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Fill one continuous carry horizon with bounded independent target segments."""
    duration = float(time[-1])
    rate = rng.uniform(*transitions_per_horizon)
    count = max(1, int(round(rate * duration / horizon_sec)))
    min_transition, max_transition = transition_sec
    min_hold, max_hold = hold_sec
    if not (count * (min_transition + min_hold) <= duration <= count * (max_transition + max_hold)):
        raise ValueError("segmented carry timing ranges cannot fill duration; change transitions-per-horizon or time bounds")
    minimum_total = count * (min_transition + min_hold)
    remaining = duration - minimum_total
    capacities = [max_transition - min_transition] * count + [max_hold - min_hold] * count
    increments = [0.0] * len(capacities)
    order = list(range(len(capacities)))
    rng.shuffle(order)
    capacity_left = float(sum(capacities))
    for position, item in enumerate(order):
        cap = capacities[item]
        capacity_left -= cap
        lower = max(0.0, remaining - capacity_left)
        upper = min(cap, remaining)
        value = upper if position == len(order) - 1 else rng.uniform(lower, upper)
        increments[item] = value
        remaining -= value
    transitions = [min_transition + increments[index] for index in range(count)]
    holds = [min_hold + increments[count + index] for index in range(count)]
    lower, upper = value_range
    current = rng.uniform(lower, upper)
    values = np.full(len(time), current, dtype=np.float64)
    cursor = 0.0
    segments = []
    for transition, hold in zip(transitions, holds):
        target = rng.uniform(lower, upper)
        start = cursor
        end = start + transition
        mask = (time >= start) & (time < end)
        if np.any(mask):
            tau = (time[mask] - start) / transition
            values[mask] = current + (target - current) * quintic_progress(tau)
        values[time >= end] = target
        hold_end = end + hold
        segments.append({
            "start_sec": float(start), "transition_sec": float(transition), "hold_sec": float(hold),
            "transition_end_sec": float(end), "hold_end_sec": float(hold_end),
            "target_start_kpa": float(current), "target_end_kpa": float(target),
        })
        current = target
        cursor = hold_end
    return values, {
        "transition_count": count,
        "transition_duration_sec": transitions,
        "hold_duration_sec": holds,
        "segments": segments,
    }


def run(args: argparse.Namespace) -> None:
    dt = 1.0 / args.freq
    horizon_steps = 512
    progress_interval = max(1, int(args.progress_every_horizons) * horizon_steps)
    time = np.arange(0.0, args.duration + 0.5 * dt, dt, dtype=np.float64)
    if args.target_mode == "segmented":
        rng = random.Random(args.seed)
        target_pos, pos_schedule = independent_segmented_target(
            time, rng, args.target_pos_range, args.pos_transitions_per_horizon,
            512.0 / args.freq, args.transition_sec, args.hold_sec,
        )
        target_neg, neg_schedule = independent_segmented_target(
            time, rng, args.target_neg_range, args.neg_transitions_per_horizon,
            512.0 / args.freq, args.transition_sec, args.hold_sec,
        )
    else:
        # Legacy target mode retained for reproducibility of existing carry profiles.
        target_pos = (
            args.target_pos
            + args.target_amp_pos * np.sin(2.0 * math.pi * args.freq_pos * time + args.phase_pos)
            + 0.35 * args.target_amp_pos * np.sin(2.0 * math.pi * args.freq_pos2 * time + 0.7 * args.phase_pos)
        )
        target_neg = (
            args.target_neg
            + args.target_amp_neg * np.sin(2.0 * math.pi * args.freq_neg * time + args.phase_neg)
            + 0.35 * args.target_amp_neg * np.sin(2.0 * math.pi * args.freq_neg2 * time + 0.7 * args.phase_neg)
        )
        pos_schedule = {"transition_count": None, "segments": []}
        neg_schedule = {"transition_count": None, "segments": []}
    dither_pos = np.sin(2.0 * math.pi * args.dither_freq_pos * time + 1.1 * args.phase_pos)
    dither_neg = np.sin(2.0 * math.pi * args.dither_freq_neg * time + 1.1 * args.phase_neg)
    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    sim.set_init_press(args.init_pos, args.init_neg)
    pressure = np.empty((len(time), 2), dtype=np.float64)
    action = np.empty((len(time), 2), dtype=np.float64)
    pressure[0] = [args.init_pos, args.init_neg]
    action[0] = [args.base_pos, args.base_neg]
    for index in range(1, len(time)):
        previous = pressure[index - 1]
        pos_error = previous[0] - target_pos[index - 1]
        neg_error = previous[1] - target_neg[index - 1]
        # Larger POS command opens the positive valve toward atmosphere and
        # lowers positive pressure.  Larger NEG command admits atmosphere and
        # raises negative pressure; hence the opposite feedback sign.
        raw_pos = args.base_pos + args.kp_pos * pos_error + args.dither_amp_pos * dither_pos[index]
        raw_neg = args.base_neg - args.kp_neg * neg_error + args.dither_amp_neg * dither_neg[index]
        # A low POS command closes the positive valve and can accumulate pump
        # pressure over hundreds of carry episodes.  This guard exists only
        # while *generating* a bounded reference; it is not part of SAC/PID.
        pos_excess = max(0.0, previous[0] - args.pos_pressure_guard_kpa)
        raw_pos += args.pos_pressure_guard_kp * pos_excess
        if previous[0] >= args.pos_pressure_guard_kpa:
            raw_pos = max(raw_pos, args.pos_pressure_guard_min_raw)
        action[index] = np.clip(
            [raw_pos, raw_neg],
            [args.raw_pos_min, args.raw_neg_min],
            [args.raw_pos_max, args.raw_neg_max],
        )
        observation, _ = sim.observe(action[index])
        pressure[index] = observation[1:3]
        if index % progress_interval == 0 or index == len(time) - 1:
            print(
                "[ INFO] carry-reference"
                f" | step | {index}/{len(time) - 1}"
                f" | horizon | {index / horizon_steps:.1f}"
                f" | P | {pressure[index, 0]:.2f},{pressure[index, 1]:.2f}"
                f" | target | {target_pos[index]:.2f},{target_neg[index]:.2f}"
                f" | raw_u | {action[index, 0]:.3f},{action[index, 1]:.3f}"
            )
    start = min(len(time) - 1, int(round(args.ignore_initial_sec * args.freq)))
    checked = pressure[start:]
    rate = np.gradient(pressure, dt, axis=0)[start:]
    feasible = bool(
        np.all(np.isfinite(pressure))
        and np.all(checked[:, 0] >= args.pos_min)
        and np.all(checked[:, 0] <= args.pos_max)
        and np.all(checked[:, 1] >= args.neg_min)
        and np.all(checked[:, 1] <= args.neg_max)
        and np.max(np.abs(rate[:, 0])) <= args.pos_rate_max
        and np.max(np.abs(rate[:, 1])) <= args.neg_rate_max
    )
    metadata = {
        "duration_sec": args.duration,
        "episode_boundaries_512": int(args.duration / (512.0 / args.freq)),
        "pressure_min_after_startup": checked.min(axis=0).tolist(),
        "pressure_max_after_startup": checked.max(axis=0).tolist(),
        "pressure_span_after_startup": np.ptp(checked, axis=0).tolist(),
        "reference_rate_abs_max_after_startup": np.max(np.abs(rate), axis=0).tolist(),
        "action_min_raw": action.min(axis=0).tolist(),
        "action_max_raw": action.max(axis=0).tolist(),
        "feasible": feasible,
    }
    if not feasible:
        raise RuntimeError(f"stabilized carry candidate is infeasible: {metadata}")

    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        time=time.astype(np.float32),
        pressure=pressure.astype(np.float32),
        raw_action=action.astype(np.float32),
        target_pressure=np.c_[target_pos, target_neg].astype(np.float32),
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
            "library": 2,
            "continuous_carry": True,
            "duration_sec": args.duration,
            "horizon_512_sec": 512.0 / args.freq,
            "episode_boundaries_inside_rollout": metadata["episode_boundaries_512"],
            "generator": "one continuous lib2 simulator rollout with bounded reference-generation feedback",
            "target_mode": args.target_mode,
            "reference_initial_pressure_kpa": [args.init_pos, args.init_neg],
            "expected_training_initial_pressure_kpa": [101.325, 101.325],
            "rate_filter_ignore_initial_sec": args.ignore_initial_sec,
            "pressure_filter_kpa": {"pos": [args.pos_min, args.pos_max], "neg": [args.neg_min, args.neg_max]},
            "rate_filter_kpa_s": [args.pos_rate_max, args.neg_rate_max],
            "parameters": {
                "target_pos": args.target_pos,
                "target_neg": args.target_neg,
                "target_amp_pos": args.target_amp_pos,
                "target_amp_neg": args.target_amp_neg,
                "freq_pos": args.freq_pos,
                "freq_neg": args.freq_neg,
                "freq_pos2": args.freq_pos2,
                "freq_neg2": args.freq_neg2,
                "kp_pos": args.kp_pos,
                "kp_neg": args.kp_neg,
                "dither_amp_pos": args.dither_amp_pos,
                "dither_amp_neg": args.dither_amp_neg,
                "raw_action_bounds": {
                    "pos": [args.raw_pos_min, args.raw_pos_max],
                    "neg": [args.raw_neg_min, args.raw_neg_max],
                },
                "pos_pressure_guard": {
                    "start_kpa": args.pos_pressure_guard_kpa,
                    "gain": args.pos_pressure_guard_kp,
                    "min_raw_action": args.pos_pressure_guard_min_raw,
                },
                "target_pos_range": list(args.target_pos_range),
                "target_neg_range": list(args.target_neg_range),
                "pos_transitions_per_horizon": list(args.pos_transitions_per_horizon),
                "neg_transitions_per_horizon": list(args.neg_transitions_per_horizon),
                "transition_sec": list(args.transition_sec),
                "hold_sec": list(args.hold_sec),
            },
            "pos_schedule": pos_schedule,
            "neg_schedule": neg_schedule,
            "validation": metadata,
            "note": "Reference-generation feedback is not used by the RL policy.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)
    rate_full = np.gradient(pressure, dt, axis=0)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(time, pressure[:, 0], "r", label="POS ref")
    axes[0].plot(time, pressure[:, 1], "b", label="NEG ref")
    axes[1].plot(time, target_pos, "r--", label="POS target")
    axes[1].plot(time, target_neg, "b--", label="NEG target")
    axes[2].plot(time, rate_full[:, 0], "r", label="dPOS/dt")
    axes[2].plot(time, rate_full[:, 1], "b", label="dNEG/dt")
    axes[3].plot(time, action[:, 0], "r", label="u POS")
    axes[3].plot(time, action[:, 1], "b", label="u NEG")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Pressure [kPa]")
    axes[1].set_ylabel("Target [kPa]")
    axes[2].set_ylabel("dP/dt [kPa/s]")
    axes[3].set_ylabel("raw action")
    axes[3].set_xlabel("Continuous time [s]")
    fig.suptitle("Long stabilized lib2 carry reference")
    fig.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] profile: {args.output}")
    print(f"[ INFO] npz: {args.npz_output}")
    print(f"[ INFO] metadata: {metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--duration", type=float, default=1024.0)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument(
        "--init-pos",
        type=float,
        default=150.0,
        help="reference-rollout initial POS pressure; carry training itself starts at atmosphere once",
    )
    parser.add_argument(
        "--init-neg",
        type=float,
        default=24.0,
        help="reference-rollout initial NEG pressure; carry training itself starts at atmosphere once",
    )
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--pos-min", type=float, default=115.0)
    parser.add_argument("--pos-max", type=float, default=220.0)
    parser.add_argument("--neg-min", type=float, default=10.0)
    parser.add_argument("--neg-max", type=float, default=30.0)
    parser.add_argument("--pos-rate-max", type=float, default=32.0)
    parser.add_argument("--neg-rate-max", type=float, default=9.5)
    parser.add_argument("--ignore-initial-sec", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=("sinusoidal", "segmented"), default="sinusoidal")
    parser.add_argument("--target-pos-range", type=parse_pair, default=parse_pair("125,190"))
    parser.add_argument("--target-neg-range", type=parse_pair, default=parse_pair("16,25"))
    parser.add_argument("--pos-transitions-per-horizon", type=parse_pair, default=parse_pair("1,3"))
    parser.add_argument("--neg-transitions-per-horizon", type=parse_pair, default=parse_pair("2,4"))
    parser.add_argument("--transition-sec", type=parse_pair, default=parse_pair("0.45,2.20"))
    parser.add_argument("--hold-sec", type=parse_pair, default=parse_pair("1.0,9.0"))
    parser.add_argument("--target-pos", type=float, default=145.0)
    parser.add_argument("--target-neg", type=float, default=20.0)
    parser.add_argument("--target-amp-pos", type=float, default=12.0)
    parser.add_argument("--target-amp-neg", type=float, default=3.0)
    parser.add_argument("--freq-pos", type=float, default=0.018)
    parser.add_argument("--freq-neg", type=float, default=0.047)
    parser.add_argument("--freq-pos2", type=float, default=0.061)
    parser.add_argument("--freq-neg2", type=float, default=0.083)
    parser.add_argument("--phase-pos", type=float, default=0.4)
    parser.add_argument("--phase-neg", type=float, default=-1.2)
    parser.add_argument("--base-pos", type=float, default=0.48)
    parser.add_argument("--base-neg", type=float, default=0.76)
    parser.add_argument("--raw-pos-min", type=float, default=0.0)
    parser.add_argument("--raw-pos-max", type=float, default=1.0)
    parser.add_argument("--raw-neg-min", type=float, default=0.0)
    parser.add_argument("--raw-neg-max", type=float, default=1.0)
    parser.add_argument("--pos-pressure-guard-kpa", type=float, default=205.0)
    parser.add_argument("--pos-pressure-guard-kp", type=float, default=0.03)
    parser.add_argument("--pos-pressure-guard-min-raw", type=float, default=0.38)
    parser.add_argument("--kp-pos", type=float, default=0.004)
    parser.add_argument("--kp-neg", type=float, default=0.004)
    parser.add_argument("--dither-amp-pos", type=float, default=0.035)
    parser.add_argument("--dither-amp-neg", type=float, default=0.025)
    parser.add_argument("--dither-freq-pos", type=float, default=0.11)
    parser.add_argument("--dither-freq-neg", type=float, default=0.17)
    parser.add_argument(
        "--progress-every-horizons",
        type=int,
        default=10,
        help="print one reference-generation status line every N 512-step horizons",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    if not (0.0 <= args.raw_pos_min <= args.raw_pos_max <= 1.0 and 0.0 <= args.raw_neg_min <= args.raw_neg_max <= 1.0):
        parser.error("raw action bounds must satisfy 0 <= min <= max <= 1")
    run(args)


if __name__ == "__main__":
    main()
