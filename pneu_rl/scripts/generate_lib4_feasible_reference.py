#!/usr/bin/env python3
"""Generate a displacement-inclusive lib4 oracle-feasible trajectory bank.

The stored RL reference is the lib4 plant output
``[actuator_pos_pressure, actuator_neg_pressure, contraction_mm]`` from a
smooth six-valve command rollout.  The command is saved only as an oracle
audit: it is never provided to the RL policy.  Candidates are replayed from
the full prefix and rejected when pressure, reserve, displacement, rate, or
terminal constraints fail, so every accepted carry block is physically
continuous in the exact tuned simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (ROOT / "pneu_env" / "src", ROOT / "pneu_ref" / "src", ROOT / "pneu_utils" / "src", SCRIPT_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from preflight_lib4 import build_sim, load_model_cfg, tuned_setup
from pneu_env.sim4 import PneuSim


STATE_NAMES = ("ch_pos", "ch_neg", "act_pos", "act_neg", "displacement_mm", "velocity_mm_s")
REFERENCE_INDICES = np.array([2, 3, 4], dtype=np.int64)


def pair(text: str) -> tuple[float, float]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != 2 or values[0] >= values[1]:
        raise argparse.ArgumentTypeError("expected an ascending pair, e.g. 100,180")
    return values


def quintic(progress: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def reset_sim(sim: PneuSim) -> None:
    sim.lib.time_reset_c()
    sim.set_init_press(
        sim.init_pos_press,
        sim.init_neg_press,
        sim.init_act_pos_press,
        sim.init_act_neg_press,
        sim.init_length,
        sim.init_velocity,
    )


def initial_state(sim: PneuSim) -> np.ndarray:
    total_length = sim.cell_length_to_total_length(sim.init_length)
    return np.array(
        [
            sim.init_pos_press,
            sim.init_neg_press,
            sim.init_act_pos_press,
            sim.init_act_neg_press,
            sim.total_length_to_contraction(total_length) * 1000.0,
            -sim.cell_to_total_displacement(sim.init_velocity) * 1000.0,
        ],
        dtype=np.float64,
    )


def simulate(sim: PneuSim, raw_action: np.ndarray) -> np.ndarray:
    """Replay a raw [0, 1] six-valve command prefix from the configured state."""
    reset_sim(sim)
    state = np.full((len(raw_action), len(STATE_NAMES)), np.nan, dtype=np.float64)
    state[0] = initial_state(sim)
    for index in range(1, len(raw_action)):
        _, info = sim.observe(raw_action[index], np.zeros(3, dtype=np.float64))
        obs = info["Observation"]
        state[index] = np.array(
            [
                obs["sen_pos"],
                obs["sen_neg"],
                obs["sen_act_pos"],
                obs["sen_act_neg"],
                sim.total_length_to_contraction(obs["sen_total_length"]) * 1000.0,
                -obs["sen_total_velocity"] * 1000.0,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(state[index])):
            break
    return state


def target_action(rng: random.Random, previous: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Sample a conflict-free actuator command and bounded chamber motion."""
    target = previous.copy()
    target[0] = rng.uniform(*args.ch_pos_raw)
    target[1] = rng.uniform(*args.ch_neg_raw)
    for lower, upper in ((2, 3), (4, 5)):
        target[lower:upper + 1] = 0.0
        mode = rng.choice(("idle", "in", "out"))
        if mode == "in":
            target[lower] = rng.uniform(*args.actuator_raw)
        elif mode == "out":
            target[upper] = rng.uniform(*args.actuator_raw)
    return target


def topology_pulse_target(previous: np.ndarray, direction: str, args: argparse.Namespace) -> np.ndarray:
    """Return a coordinated six-valve target for contraction/recovery.

    The earlier ``independent_smooth`` proposal changes each valve in
    isolation.  That is useful for a small-signal bank, but it under-excites
    lib4's coupled pressure/motion dynamics.  These two open-loop patterns
    come from the valve-flow topology used in the lib3 oracle generator:

    * ``drive``: charge the positive actuator branch and create contraction;
    * ``release``: reverse/recover the branch before the reservoirs drift.

    They are *candidate generators*, not a controller and not a restriction
    on the six policy actions during RL.  Full prefix replay plus the regular
    band/rate/reserve checks remains the feasibility decision.
    """
    if direction == "drive":
        return np.asarray(args.pulse_drive_raw, dtype=np.float64)
    if direction == "release":
        return np.asarray(args.pulse_release_raw, dtype=np.float64)
    raise ValueError(f"unknown topology pulse direction: {direction}")


def propose_block(rng: random.Random, previous: np.ndarray, args: argparse.Namespace, first: bool) -> np.ndarray:
    steps = args.horizon_steps
    time = np.arange(steps + 1, dtype=np.float64) / args.freq
    value = np.tile(previous, (steps + 1, 1))
    cursor = args.initial_hold_sec if first else 0.0
    current = previous.copy()
    while cursor < time[-1] - args.min_hold_sec:
        transition = rng.uniform(*args.transition_sec)
        hold = rng.uniform(*args.hold_sec)
        if cursor + transition > time[-1]:
            break
        if args.proposal_mode == "independent_smooth":
            target = target_action(rng, current, args)
        else:
            # Continue whichever of the two coupled states is farther from
            # the current command.  This makes the first block charge and
            # later blocks alternate rather than repeatedly selecting the
            # same valve posture at every carry boundary.
            drive = topology_pulse_target(current, "drive", args)
            release = topology_pulse_target(current, "release", args)
            if cursor <= args.initial_hold_sec + 1e-12 and first:
                direction = "drive"
            elif np.linalg.norm(current - drive) <= np.linalg.norm(current - release):
                direction = "release"
            else:
                direction = "drive"
            target = topology_pulse_target(current, direction, args)
        mask = (time >= cursor) & (time < cursor + transition)
        if np.any(mask):
            value[mask] = current + (target - current) * quintic((time[mask] - cursor) / transition)[:, None]
        value[time >= cursor + transition] = target
        current = target
        cursor += transition + hold
    return np.clip(value, 0.0, 1.0)


def candidate_metrics(state: np.ndarray, start: int, args: argparse.Namespace) -> tuple[float | None, dict[str, Any] | str]:
    checked_start = start + (int(round(args.ignore_initial_sec * args.freq)) if start == 0 else 0)
    block = state[checked_start:]
    if len(block) < 8 or not np.all(np.isfinite(block)):
        return None, "nonfinite"
    rate = np.gradient(state, 1.0 / args.freq, axis=0)[checked_start:]
    lower = np.array([args.ch_pos_band[0], args.ch_neg_band[0], args.act_pos_band[0], args.act_neg_band[0], args.disp_band[0]])
    upper = np.array([args.ch_pos_band[1], args.ch_neg_band[1], args.act_pos_band[1], args.act_neg_band[1], args.disp_band[1]])
    if np.any(block[:, :5] < lower) or np.any(block[:, :5] > upper):
        return None, "state_band"
    limits = np.array([args.ch_pos_rate_max, args.ch_neg_rate_max, args.act_pos_rate_max, args.act_neg_rate_max, args.disp_rate_max])
    if np.any(np.max(np.abs(rate[:, :5]), axis=0) > limits):
        return None, "state_rate"
    reserve = np.c_[block[:, 0] - block[:, 2], block[:, 3] - block[:, 1]]
    if np.any(reserve[:, 0] < args.min_pos_reserve) or np.any(reserve[:, 1] < args.min_neg_reserve):
        return None, "reserve"
    span = np.ptp(block[:, [2, 3, 4]], axis=0)
    if np.any(span < np.asarray(args.min_reference_span)):
        return None, "reference_span"
    if float(block[:, 4].max()) < args.min_disp_peak:
        return None, "displacement_peak"
    terminal = block[-1, :5]
    terminal_low = np.array([args.ch_pos_terminal[0], args.ch_neg_terminal[0], args.act_pos_terminal[0], args.act_neg_terminal[0], args.disp_terminal[0]])
    terminal_high = np.array([args.ch_pos_terminal[1], args.ch_neg_terminal[1], args.act_pos_terminal[1], args.act_neg_terminal[1], args.disp_terminal[1]])
    if np.any(terminal < terminal_low) or np.any(terminal > terminal_high):
        return None, "terminal_state"
    score = float(np.sum(span / np.maximum(np.asarray(args.min_reference_span), 1e-9)))
    return score, {
        "state_min": block[:, :5].min(axis=0).tolist(),
        "state_max": block[:, :5].max(axis=0).tolist(),
        "reference_span": span.tolist(),
        "displacement_peak_mm": float(block[:, 4].max()),
        "rate_abs_max": np.max(np.abs(rate[:, :5]), axis=0).tolist(),
        "reserve_min": reserve.min(axis=0).tolist(),
        "terminal_state": terminal.tolist(),
        "score": score,
    }


def generate_one(sim: PneuSim, rng: random.Random, args: argparse.Namespace, index: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    initial_action = np.array([args.ch_pos_initial, args.ch_neg_initial, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    committed = initial_action.reshape(1, 6)
    schedule: list[dict[str, Any]] = []
    for block_index in range(args.blocks):
        start = len(committed) - 1
        best: tuple[float, np.ndarray, np.ndarray, dict[str, Any]] | None = None
        rejected: Counter[str] = Counter()
        for _ in range(args.candidates_per_block):
            proposal = propose_block(rng, committed[-1], args, first=(block_index == 0))
            action = np.vstack((committed, proposal[1:]))
            state = simulate(sim, action)
            score, metrics = candidate_metrics(state, start, args)
            if score is None:
                rejected[str(metrics)] += 1
                continue
            if best is None or score > best[0]:
                best = (score, action, state, metrics)
        if best is None:
            raise RuntimeError(
                f"trajectory {index}, block {block_index}: no feasible candidate; "
                f"rejections={dict(rejected)}"
            )
        _, committed, state, metrics = best
        schedule.append(
            {
                "trajectory": index,
                "block": block_index,
                "rejections": dict(rejected),
                **metrics,
            }
        )
        print(
            f"[ INFO] lib4 oracle reference | trajectory {index + 1}/{args.bank_size} "
            f"block {block_index + 1}/{args.blocks} | ref span="
            f"{np.round(metrics['reference_span'], 3).tolist()}"
        )
    return committed, simulate(sim, committed), schedule


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cfg", default="0625_lib4_Ours")
    parser.add_argument("--state-source", choices=("model", "current_tuned"), default="current_tuned")
    parser.add_argument("--seed", type=int, default=61098)
    parser.add_argument("--bank-size", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--candidates-per-block", type=int, default=12)
    parser.add_argument(
        "--proposal-mode",
        choices=("independent_smooth", "topology_pulse"),
        default="independent_smooth",
        help="independent valve proposals or coordinated drive/release pulses",
    )
    parser.add_argument("--initial-hold-sec", type=float, default=0.8)
    parser.add_argument("--transition-sec", type=pair, default=(1.0, 3.0))
    parser.add_argument("--hold-sec", type=pair, default=(1.0, 3.0))
    parser.add_argument("--min-hold-sec", type=float, default=0.8)
    parser.add_argument("--ch-pos-initial", type=float, default=0.5)
    parser.add_argument("--ch-neg-initial", type=float, default=1.0)
    parser.add_argument("--ch-pos-raw", type=pair, default=(0.5, 1.0))
    parser.add_argument("--ch-neg-raw", type=pair, default=(0.8, 1.0))
    parser.add_argument("--actuator-raw", type=pair, default=(0.15, 0.85))
    parser.add_argument(
        "--pulse-drive-raw",
        type=lambda text: tuple(float(value) for value in text.split(",")),
        default=(0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
        help="six raw commands for coordinated contraction drive",
    )
    parser.add_argument(
        "--pulse-release-raw",
        type=lambda text: tuple(float(value) for value in text.split(",")),
        default=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        help="six raw commands for coordinated recovery/release",
    )
    parser.add_argument("--ch-pos-band", type=pair, default=(100.0, 200.0))
    parser.add_argument("--ch-neg-band", type=pair, default=(1.0, 105.0))
    parser.add_argument("--act-pos-band", type=pair, default=(90.0, 160.0))
    parser.add_argument("--act-neg-band", type=pair, default=(20.0, 110.0))
    # The current tuned lib4 state starts near 5.2 mm contraction and the
    # first oracle sweep reaches the 0 mm extension stop.  The historical
    # 7--21 mm mapping is therefore not a feasible initial training envelope.
    parser.add_argument("--disp-band", type=pair, default=(0.0, 32.0))
    parser.add_argument("--ch-pos-terminal", type=pair, default=(100.0, 195.0))
    parser.add_argument("--ch-neg-terminal", type=pair, default=(1.0, 100.0))
    parser.add_argument("--act-pos-terminal", type=pair, default=(90.0, 155.0))
    parser.add_argument("--act-neg-terminal", type=pair, default=(20.0, 108.0))
    parser.add_argument("--disp-terminal", type=pair, default=(0.0, 31.0))
    parser.add_argument("--ch-pos-rate-max", type=float, default=80.0)
    parser.add_argument("--ch-neg-rate-max", type=float, default=80.0)
    parser.add_argument("--act-pos-rate-max", type=float, default=55.0)
    parser.add_argument("--act-neg-rate-max", type=float, default=45.0)
    parser.add_argument("--disp-rate-max", type=float, default=20.0)
    parser.add_argument("--min-pos-reserve", type=float, default=3.0)
    parser.add_argument("--min-neg-reserve", type=float, default=3.0)
    parser.add_argument("--min-reference-span", type=lambda text: tuple(float(v) for v in text.split(",")), default=(1.0, 1.0, 0.1))
    parser.add_argument(
        "--min-disp-peak",
        type=float,
        default=0.0,
        help="minimum contraction peak per accepted block after startup [mm]",
    )
    parser.add_argument("--ignore-initial-sec", type=float, default=1.0)
    parser.add_argument("--npz-output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.min_reference_span) != 3 or any(value <= 0.0 for value in args.min_reference_span):
        parser.error("--min-reference-span must be three positive values: pos,neg,displacement")
    if (
        len(args.pulse_drive_raw) != 6
        or len(args.pulse_release_raw) != 6
        or any(not 0.0 <= value <= 1.0 for value in (*args.pulse_drive_raw, *args.pulse_release_raw))
    ):
        parser.error("pulse drive/release commands must each contain six values in [0, 1]")
    if args.min_disp_peak < 0.0:
        parser.error("--min-disp-peak must be non-negative")
    if args.bank_size < 1 or args.blocks < 1 or args.horizon_steps < 8 or args.candidates_per_block < 1:
        parser.error("bank-size, blocks, candidates must be positive and horizon must be >= 8")

    cfg, cfg_path = load_model_cfg(args.model_cfg)
    sim, sim_record = build_sim(cfg, args.state_source)
    if not math.isclose(sim.freq, args.freq, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"model frequency {sim.freq} differs from --freq {args.freq}")
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    schedules: list[list[dict[str, Any]]] = []
    for index in range(args.bank_size):
        rng = random.Random(args.seed + 10007 * index)
        action, state, schedule = generate_one(sim, rng, args, index)
        actions.append(action)
        states.append(state)
        schedules.append(schedule)

    raw_action = np.stack(actions)
    state = np.stack(states)
    reference = state[:, :, REFERENCE_INDICES]
    time = np.arange(raw_action.shape[1], dtype=np.float64) / args.freq
    physical_action = sim.scale_low + (sim.scale_high - sim.scale_low) * raw_action
    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        time=np.repeat(time[None, :], args.bank_size, axis=0).astype(np.float32),
        reference=reference.astype(np.float32),
        state=state.astype(np.float32),
        raw_action=raw_action.astype(np.float32),
        physical_action=physical_action.astype(np.float32),
    )
    profile = {
        "kind": "trajectory_bank_nd",
        "kwargs": {
            "trajectory_npz": relative_to_root(args.npz_output),
            "reference_key": "reference",
            "seed": args.seed,
            "sample_mode": "coverage",
            "repeat_after_end": False,
        },
        "design": {
            "library": 4,
            "reference_channels": ["act_pos_kpa", "act_neg_kpa", "contraction_mm"],
            "continuous_carry": True,
            "bank_size": args.bank_size,
            "block_steps": args.horizon_steps,
            "blocks_per_hard_reset": args.blocks,
            "frequency_hz": args.freq,
            "duration_sec": float(time[-1]),
            "oracle_feasible": True,
            "oracle_action_is_policy_input": False,
            "proposal_mode": args.proposal_mode,
            "pulse_patterns_raw": {
                "drive": list(args.pulse_drive_raw),
                "release": list(args.pulse_release_raw),
            } if args.proposal_mode == "topology_pulse" else None,
            "model_cfg": relative_to_root(cfg_path),
            "simulator": sim_record,
            "filters": {
                "state_bands": {
                    "ch_pos": list(args.ch_pos_band), "ch_neg": list(args.ch_neg_band),
                    "act_pos": list(args.act_pos_band), "act_neg": list(args.act_neg_band),
                    "displacement_mm": list(args.disp_band),
                },
                "terminal_bands": {
                    "ch_pos": list(args.ch_pos_terminal), "ch_neg": list(args.ch_neg_terminal),
                    "act_pos": list(args.act_pos_terminal), "act_neg": list(args.act_neg_terminal),
                    "displacement_mm": list(args.disp_terminal),
                },
                "rate_abs_max": [args.ch_pos_rate_max, args.ch_neg_rate_max, args.act_pos_rate_max, args.act_neg_rate_max, args.disp_rate_max],
                "minimum_reserve_kpa": [args.min_pos_reserve, args.min_neg_reserve],
                "minimum_reference_span": list(args.min_reference_span),
                "minimum_displacement_peak_mm": args.min_disp_peak,
                "ignore_initial_sec": args.ignore_initial_sec,
            },
            "schedule": [item for schedule in schedules for item in schedule],
        },
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    with args.profile_output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)
    print(json.dumps({"profile": str(args.profile_output), "npz": str(args.npz_output), "reference_shape": list(reference.shape)}, indent=2))


if __name__ == "__main__":
    main()
