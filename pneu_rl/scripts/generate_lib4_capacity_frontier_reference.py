#!/usr/bin/env python3
"""Generate a displacement-targeted, capacity-screened lib4 reference bank.

Unlike the fixed R2 drive/release pulse, this generator first assigns a
desired contraction frontier to each bank member.  It then samples smooth,
conflict-free valve programs and projects that output request onto the set of
programs that pass a full tuned-simulator replay.  The accepted training file
contains only the resulting pressure-motion reference; stored actions remain
offline feasibility audit data.

Two tiers are deliberately separate:

* ``core`` stays inside the contraction range covered by the tuning data;
* ``edge`` explores the nominal model beyond that data and is not a training
  source until parameter/initial-state robustness and real checks are added.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (ROOT / "pneu_env" / "src", ROOT / "pneu_ref" / "src", ROOT / "pneu_utils" / "src", SCRIPT_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from generate_lib4_feasible_reference import REFERENCE_INDICES, STATE_NAMES, initial_state, quintic, reset_sim, simulate
from preflight_lib4 import build_sim, load_model_cfg


MEASURED_CONTRACTION_MAX_MM = 21.467442133834744
INTENDED_STROKE_MAX_MM = 32.0

CONTRACT_POSTURES = (
    (0.0, 0.5, 1.0, 0.0, 0.0, 1.0),
    (0.5, 1.0, 1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 0.0, 0.0, 1.0),
    (0.5, 0.5, 1.0, 0.0, 0.0, 1.0),
)
RECOVER_POSTURES = (
    (0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    (0.5, 0.5, 0.0, 1.0, 1.0, 0.0),
    (1.0, 0.5, 0.0, 1.0, 1.0, 0.0),
    (0.0, 0.5, 0.0, 0.0, 0.8, 0.0),
)


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def scaled_posture(posture: tuple[float, ...], strength: float) -> np.ndarray:
    value = np.asarray(posture, dtype=np.float64).copy()
    value[2:] *= float(strength)
    return value


def render_program(
    rng: random.Random,
    *,
    total_steps: int,
    freq: float,
    desired_peak_mm: float,
    tier: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Render a smooth contract/recover/contract output-request program."""
    duration = total_steps / freq
    time = np.arange(total_steps + 1, dtype=np.float64) / freq
    initial = np.array([0.5, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    action = np.tile(initial, (total_steps + 1, 1))
    cursor = rng.uniform(0.35, 0.75)
    current = initial.copy()
    events: list[dict[str, Any]] = []

    if tier == "core":
        nominal_strength = float(
            np.interp(
                desired_peak_mm,
                (8.0, MEASURED_CONTRACTION_MAX_MM),
                (0.63, 0.84),
            )
        )
        strength = float(np.clip(nominal_strength + rng.uniform(-0.045, 0.045), 0.56, 0.88))
        first_contract_total = np.interp(desired_peak_mm, (8.0, MEASURED_CONTRACTION_MAX_MM), (6.0, 15.5))
        recover_total = rng.uniform(7.0, 10.5)
    else:
        nominal_strength = float(
            np.interp(
                desired_peak_mm,
                (MEASURED_CONTRACTION_MAX_MM, 31.0),
                (0.90, 1.0),
            )
        )
        strength = float(np.clip(nominal_strength + rng.uniform(-0.025, 0.025), 0.88, 1.0))
        first_contract_total = np.interp(desired_peak_mm, (MEASURED_CONTRACTION_MAX_MM, 31.0), (13.0, 22.0))
        recover_total = rng.uniform(6.0, 9.0)
    first_contract_total *= rng.uniform(0.88, 1.12)

    if tier == "edge":
        # Edge members are nominal frontier probes, not cyclic training
        # tasks.  A single slow approach avoids conflating extrapolated stroke
        # reachability with an unverified high-rate recovery maneuver.
        phases = [("contract", max(duration - cursor, 0.0))]
    else:
        phases = [
            ("contract", first_contract_total),
            ("recover", recover_total),
            ("contract", max(duration, 0.0)),
        ]
    for phase_index, (direction, phase_total) in enumerate(phases):
        remaining = duration - cursor
        if remaining <= 0.5:
            break
        phase_total = min(float(phase_total), remaining)
        transition_bounds = (3.0, 5.0) if tier == "core" else (10.0, 16.0)
        transition = min(rng.uniform(*transition_bounds), max(phase_total - 0.25, 0.25))
        if direction == "contract":
            posture = rng.choice(CONTRACT_POSTURES) if tier == "core" else CONTRACT_POSTURES[0]
            phase_strength = strength * rng.uniform(0.94, 1.0)
        else:
            if tier == "core":
                posture = rng.choice(RECOVER_POSTURES)
                phase_strength = rng.uniform(0.55, min(0.9, strength + 0.05))
            else:
                # A neutral recovery is slower than commanding the inverse
                # actuator topology and avoids manufacturing high-rate edge
                # trajectories merely to reach an extrapolated stroke bin.
                posture = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                phase_strength = 1.0
        target = scaled_posture(posture, phase_strength)
        transition_end = cursor + transition
        phase_end = cursor + phase_total
        mask = (time >= cursor) & (time < transition_end)
        if np.any(mask):
            action[mask] = current + (target - current) * quintic((time[mask] - cursor) / transition)[:, None]
        action[time >= transition_end] = target
        events.append(
            {
                "phase": phase_index,
                "direction": direction,
                "start_sec": cursor,
                "transition_sec": transition,
                "end_sec": phase_end,
                "target_raw": target,
            }
        )
        current = target
        cursor = phase_end
    return np.clip(action, 0.0, 1.0), events


def apply_edge_state_gate(
    sim: Any,
    action: np.ndarray,
    rng: random.Random,
    *,
    desired_peak_mm: float,
    freq: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an offline state-gated edge command.

    Chamber-pressure management and displacement stopping are deliberately
    separate.  A hysteretic chamber command preserves actuator authority
    without continuously charging the reservoir, while the four actuator
    valves taper only after the desired stroke frontier is approached.  The
    resulting command is independently replayed from reset before acceptance;
    neither the oracle command nor these gates are policy inputs.
    """
    reset_sim(sim)
    gated = action.copy()
    state = initial_state(sim)
    chamber_high = rng.uniform(190.0, 196.0)
    chamber_low = rng.uniform(162.0, 178.0)
    displacement_gate_mm = desired_peak_mm - rng.uniform(-0.1, 0.6)
    chamber_raw_rate = rng.uniform(0.7, 1.0) / freq
    actuator_raw_rate = rng.uniform(0.12, 0.24) / freq
    chamber_target = float(gated[0, 0])
    actuator_stopping = False
    chamber_switches = 0
    displacement_gate_index: int | None = None

    for index in range(1, len(gated)):
        if state[0] >= chamber_high and chamber_target != 1.0:
            chamber_target = 1.0
            chamber_switches += 1
        elif state[0] <= chamber_low and chamber_target != 0.0:
            chamber_target = 0.0
            chamber_switches += 1
        previous_chamber_raw = gated[index - 1, 0]
        gated[index, 0] = previous_chamber_raw + np.clip(
            chamber_target - previous_chamber_raw,
            -chamber_raw_rate,
            chamber_raw_rate,
        )

        if state[4] >= displacement_gate_mm and not actuator_stopping:
            actuator_stopping = True
            displacement_gate_index = index
        if actuator_stopping:
            previous_actuator_raw = gated[index - 1, 2:6]
            gated[index, 2:6] = previous_actuator_raw + np.clip(
                -previous_actuator_raw,
                -actuator_raw_rate,
                actuator_raw_rate,
            )

        _, info = sim.observe(gated[index], np.zeros(3, dtype=np.float64))
        obs = info["Observation"]
        state = np.array(
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
    return np.clip(gated, 0.0, 1.0), {
        "kind": "offline_state_gate",
        "gate_is_policy_input": False,
        "ch_pos_hysteresis_kpa": [chamber_low, chamber_high],
        "chamber_raw_rate_per_sec": chamber_raw_rate * freq,
        "chamber_switch_count": chamber_switches,
        "displacement_gate_mm": displacement_gate_mm,
        "displacement_gate_sec": (
            None if displacement_gate_index is None else displacement_gate_index / freq
        ),
        "actuator_raw_taper_rate_per_sec": actuator_raw_rate * freq,
    }


def feature_bins(state: np.ndarray, freq: float) -> set[tuple[int, int, int, int]]:
    velocity = state[:, 5]
    signs = np.where(velocity > 0.5, 1, np.where(velocity < -0.5, -1, 0))
    stride = max(int(round(freq / 5.0)), 1)
    return {
        (
            int(np.floor(row[2] / 8.0)),
            int(np.floor(row[3] / 8.0)),
            int(np.floor(row[4] / 4.0)),
            int(sign),
        )
        for row, sign in zip(state[::stride], signs[::stride])
    }


def candidate_metrics(
    state: np.ndarray,
    *,
    freq: float,
    desired_peak_mm: float,
    tier: str,
) -> tuple[float | None, dict[str, Any] | str, set[tuple[int, int, int, int]]]:
    ignore = int(round(freq))
    checked = state[ignore:]
    if len(checked) < 8 or not np.all(np.isfinite(checked)):
        return None, "nonfinite", set()
    rate = np.gradient(state, 1.0 / freq, axis=0)[ignore:]
    lower = np.array([100.0, 1.0, 90.0, 15.0, 0.0])
    # The measured tuning run reaches 213.12 kPa on chamber+, so edge probes
    # use a small 215 kPa data-backed guard while core training remains at the
    # more conservative 210 kPa bound.
    ch_pos_upper = 210.0 if tier == "core" else 215.0
    upper = np.array([ch_pos_upper, 105.0, 165.0, 115.0, INTENDED_STROKE_MAX_MM])
    if np.any(checked[:, :5] < lower) or np.any(checked[:, :5] > upper):
        low_margin = np.min(checked[:, :5] - lower, axis=0)
        high_margin = np.min(upper - checked[:, :5], axis=0)
        worst_index = int(np.argmin(np.minimum(low_margin, high_margin)))
        return None, f"state_band:{STATE_NAMES[worst_index]}", set()
    rate_limits = np.array([80.0, 80.0, 55.0, 45.0, 20.0])
    max_rate = np.max(np.abs(rate[:, :5]), axis=0)
    if np.any(max_rate > rate_limits):
        worst_index = int(np.argmax(max_rate / rate_limits))
        return None, f"state_rate:{STATE_NAMES[worst_index]}", set()
    reserve = np.c_[checked[:, 0] - checked[:, 2], checked[:, 3] - checked[:, 1]]
    if np.any(reserve.min(axis=0) < np.array([3.0, 3.0])):
        return None, "reserve", set()
    reference = checked[:, REFERENCE_INDICES]
    span = np.ptp(reference, axis=0)
    if np.any(span < np.array([1.0, 1.0, 2.0])):
        return None, "reference_span", set()
    peak = float(np.max(reference[:, 2]))
    if tier == "core" and peak > MEASURED_CONTRACTION_MAX_MM + 0.25:
        return None, "outside_measured_core", set()
    if tier == "edge" and peak <= MEASURED_CONTRACTION_MAX_MM:
        return None, "no_frontier_gain", set()
    bins = feature_bins(checked, freq)
    target_error = abs(peak - desired_peak_mm)
    target_error_limit = 4.5 if tier == "core" else 6.0
    if target_error > target_error_limit:
        return None, "target_projection_error", set()
    score = float(0.5 * len(bins) + span[2] + 0.05 * span[0] + 0.05 * span[1] - 20.0 * target_error)
    return score, {
        "desired_peak_mm": desired_peak_mm,
        "achieved_peak_mm": peak,
        "target_error_mm": target_error,
        "state_min": checked[:, :5].min(axis=0),
        "state_max": checked[:, :5].max(axis=0),
        "reference_span": span,
        "rate_abs_max": max_rate,
        "reserve_min": reserve.min(axis=0),
        "terminal_state": checked[-1, :5],
        "feature_bin_count": len(bins),
        "score": score,
    }, bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cfg", default="0625_lib4_Ours")
    parser.add_argument("--state-source", choices=("model", "current_tuned"), default="current_tuned")
    parser.add_argument("--tier", choices=("core", "edge"), default="core")
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--bank-size", type=int, default=16)
    parser.add_argument("--candidates-per-member", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--desired-min-mm", type=float, default=None)
    parser.add_argument("--desired-max-mm", type=float, default=None)
    parser.add_argument(
        "--reuse-selection-profile",
        type=Path,
        default=None,
        help="rebuild previously selected candidate seeds without repeating candidate search",
    )
    parser.add_argument("--npz-output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    args = parser.parse_args()
    if args.bank_size < 1 or args.candidates_per_member < 1 or args.blocks != 4 or args.block_steps != 512:
        parser.error("bank/candidates must be positive; this protocol requires 4 blocks of 512 steps")

    cfg, cfg_path = load_model_cfg(args.model_cfg)
    sim, sim_record = build_sim(cfg, args.state_source)
    if not math.isclose(float(sim.freq), args.freq, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("simulator/reference frequency mismatch")
    total_steps = args.blocks * args.block_steps
    if args.tier == "core":
        desired_min = 8.0 if args.desired_min_mm is None else args.desired_min_mm
        desired_max = MEASURED_CONTRACTION_MAX_MM - 0.25 if args.desired_max_mm is None else args.desired_max_mm
    else:
        desired_min = MEASURED_CONTRACTION_MAX_MM + 0.75 if args.desired_min_mm is None else args.desired_min_mm
        desired_max = 30.5 if args.desired_max_mm is None else args.desired_max_mm
    if not 0.0 <= desired_min < desired_max <= INTENDED_STROKE_MAX_MM:
        parser.error("desired range must be ascending and lie inside the 0..32 mm intended stroke")
    desired = np.linspace(desired_min, desired_max, args.bank_size)

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    schedules: list[dict[str, Any]] = []
    global_bins: set[tuple[int, int, int, int]] = set()
    master = random.Random(args.seed)
    reused_schedule: list[dict[str, Any]] | None = None
    if args.reuse_selection_profile is not None:
        reused_profile = yaml.safe_load(args.reuse_selection_profile.read_text(encoding="utf-8"))
        reused_schedule = list(reused_profile["design"]["schedule"])
        if len(reused_schedule) != args.bank_size:
            raise ValueError("reuse profile bank size does not match --bank-size")
    for member, desired_peak in enumerate(desired):
        if reused_schedule is not None:
            old = reused_schedule[member]
            candidate_seed = int(old["candidate_seed"])
            desired_peak = float(old["desired_peak_mm"])
            action, events = render_program(
                random.Random(candidate_seed),
                total_steps=total_steps,
                freq=args.freq,
                desired_peak_mm=desired_peak,
                tier=args.tier,
            )
            if args.tier == "edge":
                action, gate = apply_edge_state_gate(
                    sim,
                    action,
                    random.Random(candidate_seed ^ 0x45D9F3B),
                    desired_peak_mm=desired_peak,
                    freq=args.freq,
                )
                events.append(gate)
            state = simulate(sim, action)
            score, metrics, bins = candidate_metrics(
                state,
                freq=args.freq,
                desired_peak_mm=desired_peak,
                tier=args.tier,
            )
            if score is None:
                raise RuntimeError(f"reused member {member} no longer passes: {metrics}")
            coverage_gain = len(bins - global_bins)
            global_bins.update(bins)
            record = dict(metrics)
            record["coverage_gain"] = coverage_gain
            record["selection_score"] = float(score + 1.5 * coverage_gain)
            actions.append(action)
            states.append(state)
            schedules.append(
                {
                    "trajectory": member,
                    "tier": args.tier,
                    "candidate_seed": candidate_seed,
                    "rejections": dict(old.get("rejections", {})),
                    **record,
                    "events": events,
                }
            )
            print(
                f"[ INFO] lib4 capacity reference | reused {args.tier} member {member + 1}/{args.bank_size} "
                f"| desired={desired_peak:.2f} achieved={record['achieved_peak_mm']:.2f} mm"
            )
            continue
        best: tuple[float, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], set[tuple[int, int, int, int]], int] | None = None
        rejections: dict[str, int] = {}
        for _ in range(args.candidates_per_member):
            candidate_seed = master.randrange(1, 2**31 - 1)
            rng = random.Random(candidate_seed)
            action, events = render_program(
                rng,
                total_steps=total_steps,
                freq=args.freq,
                desired_peak_mm=float(desired_peak),
                tier=args.tier,
            )
            if args.tier == "edge":
                action, gate = apply_edge_state_gate(
                    sim,
                    action,
                    rng,
                    desired_peak_mm=float(desired_peak),
                    freq=args.freq,
                )
                events.append(gate)
            state = simulate(sim, action)
            score, metrics, bins = candidate_metrics(
                state,
                freq=args.freq,
                desired_peak_mm=float(desired_peak),
                tier=args.tier,
            )
            if score is None:
                reason = str(metrics)
                rejections[reason] = rejections.get(reason, 0) + 1
                continue
            coverage_gain = len(bins - global_bins)
            total_score = float(score + 1.5 * coverage_gain)
            record = dict(metrics)
            record["coverage_gain"] = coverage_gain
            record["selection_score"] = total_score
            if best is None or total_score > best[0]:
                best = (total_score, action, state, record, events, bins, candidate_seed)
        if best is None:
            raise RuntimeError(f"member {member}: no {args.tier} candidate passed; rejections={rejections}")
        _, action, state, metrics, events, bins, candidate_seed = best
        global_bins.update(bins)
        actions.append(action)
        states.append(state)
        schedules.append(
            {
                "trajectory": member,
                "tier": args.tier,
                "candidate_seed": candidate_seed,
                "rejections": rejections,
                **metrics,
                "events": events,
            }
        )
        print(
            f"[ INFO] lib4 capacity reference | {args.tier} member {member + 1}/{args.bank_size} "
            f"| desired={desired_peak:.2f} achieved={metrics['achieved_peak_mm']:.2f} mm "
            f"| bins+={metrics['coverage_gain']}"
        )

    raw_action = np.stack(actions)
    state = np.stack(states)
    reference = state[:, :, REFERENCE_INDICES]
    time_1d = np.arange(total_steps + 1, dtype=np.float64) / args.freq
    time = np.repeat(time_1d[None, :], args.bank_size, axis=0)
    physical_action = sim.scale_low + (sim.scale_high - sim.scale_low) * raw_action
    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        time=time.astype(np.float32),
        reference=reference.astype(np.float32),
        state=state.astype(np.float32),
        # Oracle commands are audit data, not policy inputs.  Preserve them in
        # float64 because a 40.96 s nonlinear replay can amplify float32 valve
        # quantization enough to obscure an otherwise exact certificate.
        raw_action=raw_action.astype(np.float64),
        physical_action=physical_action.astype(np.float64),
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
            "method": "displacement-targeted sampled projection onto full-replay admissible outputs",
            "tier": args.tier,
            "training_eligible": args.tier == "core",
            "edge_requires_robust_and_real_gate": args.tier == "edge",
            "reference_channels": ["act_pos_kpa", "act_neg_kpa", "contraction_mm"],
            "oracle_action_is_policy_input": False,
            "oracle_action_storage_dtype": "float64",
            "continuous_carry": True,
            "bank_size": args.bank_size,
            "block_steps": args.block_steps,
            "blocks_per_hard_reset": args.blocks,
            "frequency_hz": args.freq,
            "duration_sec": float(time_1d[-1]),
            "model_cfg": relative_to_root(cfg_path),
            "simulator": sim_record,
            "support": {
                "intended_stroke_mm": [0.0, INTENDED_STROKE_MAX_MM],
                "tuning_data_contraction_max_mm": MEASURED_CONTRACTION_MAX_MM,
                "desired_peak_range_mm": [desired_min, desired_max],
                "meaning": "core is data-supported nominal feasibility; edge is nominal extrapolation only",
            },
            "filters": {
                "state_bands": {
                    "ch_pos": [100.0, 210.0 if args.tier == "core" else 215.0],
                    "ch_neg": [1.0, 105.0],
                    "act_pos": [90.0, 165.0],
                    "act_neg": [15.0, 115.0],
                    "displacement_mm": [0.0, INTENDED_STROKE_MAX_MM],
                },
                "terminal_bands": {
                    "ch_pos": [100.0, 210.0 if args.tier == "core" else 215.0],
                    "ch_neg": [1.0, 105.0],
                    "act_pos": [90.0, 165.0],
                    "act_neg": [15.0, 115.0],
                    "displacement_mm": [0.0, INTENDED_STROKE_MAX_MM],
                },
                "rate_abs_max": [80.0, 80.0, 55.0, 45.0, 20.0],
                "minimum_reserve_kpa": [3.0, 3.0],
                "minimum_reference_span": [1.0, 1.0, 2.0],
                "maximum_target_projection_error_mm": 4.5 if args.tier == "core" else 6.0,
                "ignore_initial_sec": 1.0,
            },
            "coverage_feature": "(act_pos/8kPa, act_neg/8kPa, contraction/4mm, velocity_sign)",
            "global_feature_bin_count": len(global_bins),
            "schedule": schedules,
        },
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(yaml.safe_dump(serializable(profile), sort_keys=False), encoding="utf-8")
    print(
        yaml.safe_dump(
            {
                "profile": str(args.profile_output),
                "npz": str(args.npz_output),
                "npz_sha256": hashlib.sha256(args.npz_output.read_bytes()).hexdigest(),
                "reference_shape": list(reference.shape),
                "contraction_min_mm": float(np.min(reference[:, :, 2])),
                "contraction_max_mm": float(np.max(reference[:, :, 2])),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
