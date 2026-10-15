#!/usr/bin/env python3
"""Run lib3 matched-pulse tests for reserve and history-dependent action effect.

This script is a simulator diagnostic, not RL training.  It performs two
paired counterfactual experiments:

1. Reserve sweep: start from low/mid/high chamber--actuator headroom, replay an
   identical preparation sequence, and compare an identical valve pulse with
   a no-pulse control.  Atmosphere-connected actuator valves are included as a
   negative-control probe whose direct pressure path does not use chamber
   reserve.
2. History matching: generate many dynamically prepared states, find pairs
   with nearly equal current four-pressure observations but opposite target
   pressure rates, and apply the same pulse and its no-pulse control.  This
   tests whether recent dynamics change the short-horizon action effect after
   current pressure has been approximately matched.

The simulator's valve z/area/q_pred/mdot diagnostics are saved for mechanism
analysis only.  They are never proposed as real-policy observations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim3 import PneuSim


ATM = 101.325
FREQ_HZ = 50.0
DT = 1.0 / FREQ_HZ
PHYSICAL_LOW = 0.8
PHYSICAL_HIGH = 1.0
PRESSURE_NAMES = ("ch_pos", "ch_neg", "act_pos", "act_neg")
VALVE_NAMES = (
    "ch_pos",
    "ch_neg",
    "act_pos_in",
    "act_pos_out",
    "act_neg_in",
    "act_neg_out",
)


@dataclass
class Trajectory:
    pre_pressure: np.ndarray
    pre_rate: np.ndarray
    pre_reserve: np.ndarray
    pre_debug: dict[str, float]
    response_pressure: np.ndarray
    response_debug: list[dict[str, float]]


@dataclass
class Candidate:
    candidate_id: int
    branch: str
    actions: np.ndarray
    pressure: np.ndarray
    rate: np.ndarray
    reserve: np.ndarray
    action_delta_norm: float


def parse_csv_floats(text: str) -> np.ndarray:
    values = np.asarray([float(item) for item in text.split(",")], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid numeric list: {text!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "pneu_rl/exp/lib3_matched_pulse_observability/"
            "260824_reserve_history_pulse"
        ),
    )
    parser.add_argument("--seed", type=int, default=73109)
    parser.add_argument("--reserve-levels", default="5,15,30,45")
    parser.add_argument("--pos-act-levels", default="112,125,138")
    parser.add_argument("--neg-act-levels", default="62,75,88")
    parser.add_argument("--prep-steps", type=int, default=15)
    parser.add_argument("--response-steps", type=int, default=30)
    parser.add_argument("--pulse-physical", type=float, default=0.95)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument("--history-candidates", type=int, default=1200)
    parser.add_argument("--history-prep-steps", type=int, default=30)
    parser.add_argument("--history-pairs", type=int, default=8)
    parser.add_argument("--pressure-match-max-kpa", type=float, default=2.5)
    parser.add_argument("--reserve-match-max-kpa", type=float, default=2.5)
    parser.add_argument("--minimum-opposite-rate-kpa-s", type=float, default=0.5)
    parser.add_argument("--maximum-opposite-rate-kpa-s", type=float, default=50.0)
    return parser.parse_args()


def make_sim() -> PneuSim:
    return PneuSim(
        freq=FREQ_HZ,
        delay=0.1,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=PHYSICAL_LOW,
        scale_high=PHYSICAL_HIGH,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )


def physical_to_raw(action: np.ndarray | float) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    return np.clip(
        (value - PHYSICAL_LOW) / (PHYSICAL_HIGH - PHYSICAL_LOW),
        0.0,
        1.0,
    )


def constant_action(physical: float = PHYSICAL_LOW) -> np.ndarray:
    return physical_to_raw(np.full(6, physical, dtype=np.float64))


def reserve_from_pressure(pressure: np.ndarray) -> np.ndarray:
    p = np.asarray(pressure, dtype=np.float64)
    return np.asarray([p[0] - p[2], p[3] - p[1]], dtype=np.float64)


def run_trajectory(
    sim: PneuSim,
    initial_pressure: tuple[float, float, float, float],
    prep_actions: np.ndarray,
    response_action: np.ndarray,
    response_steps: int,
) -> Trajectory:
    sim.set_init_press(*initial_pressure)
    pre_pressure_rows: list[np.ndarray] = []
    for action in prep_actions:
        obs, _ = sim.observe(np.asarray(action, dtype=np.float64))
        pre_pressure_rows.append(np.asarray(obs[1:5], dtype=np.float64).copy())
    if len(pre_pressure_rows) < 2:
        raise ValueError("at least two preparation steps are required")
    pre_pressure_array = np.asarray(pre_pressure_rows)
    pre_pressure = pre_pressure_array[-1]
    pre_rate = (pre_pressure_array[-1] - pre_pressure_array[-2]) * FREQ_HZ
    pre_debug = {key: float(value) for key, value in sim.get_valve_debug().items()}

    response_pressure = [pre_pressure.copy()]
    response_debug: list[dict[str, float]] = [pre_debug.copy()]
    for _ in range(response_steps):
        obs, _ = sim.observe(np.asarray(response_action, dtype=np.float64))
        response_pressure.append(np.asarray(obs[1:5], dtype=np.float64).copy())
        response_debug.append(
            {key: float(value) for key, value in sim.get_valve_debug().items()}
        )
    return Trajectory(
        pre_pressure=pre_pressure,
        pre_rate=pre_rate,
        pre_reserve=reserve_from_pressure(pre_pressure),
        pre_debug=pre_debug,
        response_pressure=np.asarray(response_pressure),
        response_debug=response_debug,
    )


def paired_counterfactual(
    sim: PneuSim,
    initial_pressure: tuple[float, float, float, float],
    prep_actions: np.ndarray,
    pulse_action: np.ndarray,
    response_steps: int,
) -> tuple[Trajectory, Trajectory]:
    control = run_trajectory(
        sim,
        initial_pressure,
        prep_actions,
        constant_action(),
        response_steps,
    )
    pulse = run_trajectory(
        sim,
        initial_pressure,
        prep_actions,
        pulse_action,
        response_steps,
    )
    mismatch = float(np.max(np.abs(control.pre_pressure - pulse.pre_pressure)))
    if mismatch > 1e-10:
        raise RuntimeError(f"paired replay is not deterministic: pre mismatch={mismatch}")
    return pulse, control


def pulse_action(valve_index: int, physical: float) -> np.ndarray:
    action = np.full(6, PHYSICAL_LOW, dtype=np.float64)
    action[valve_index] = physical
    return physical_to_raw(action)


def counterfactual_effect(
    pulse: Trajectory,
    control: Trajectory,
    horizon: int,
) -> np.ndarray:
    pulse_delta = pulse.response_pressure[horizon] - pulse.response_pressure[0]
    control_delta = control.response_pressure[horizon] - control.response_pressure[0]
    return pulse_delta - control_delta


def integrated_mdot_effect(
    pulse: Trajectory,
    control: Trajectory,
    valve_name: str,
    horizon: int,
) -> float:
    key = f"{valve_name}_mdot"
    pulse_values = np.asarray(
        [row[key] for row in pulse.response_debug[1 : horizon + 1]], dtype=np.float64
    )
    control_values = np.asarray(
        [row[key] for row in control.response_debug[1 : horizon + 1]], dtype=np.float64
    )
    return float(np.sum(pulse_values - control_values) * DT)


def reserve_conditions(
    reserve_levels: np.ndarray,
    pos_levels: np.ndarray,
    neg_levels: np.ndarray,
) -> Iterable[tuple[str, float, float, tuple[float, float, float, float]]]:
    for actuator in pos_levels:
        for reserve in reserve_levels:
            chamber = actuator + reserve
            if chamber <= 220.0:
                yield "positive", float(actuator), float(reserve), (
                    float(chamber),
                    35.0,
                    float(actuator),
                    70.0,
                )
    for actuator in neg_levels:
        for reserve in reserve_levels:
            chamber = actuator - reserve
            if chamber >= 10.0:
                yield "negative", float(actuator), float(reserve), (
                    160.0,
                    float(chamber),
                    125.0,
                    float(actuator),
                )


def run_reserve_sweep(
    args: argparse.Namespace,
    sim: PneuSim,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reserve_levels = parse_csv_floats(args.reserve_levels)
    pos_levels = parse_csv_floats(args.pos_act_levels)
    neg_levels = parse_csv_floats(args.neg_act_levels)
    prep = np.tile(constant_action(), (args.prep_steps, 1))
    summary_rows: list[dict[str, float | str]] = []
    trace_rows: list[dict[str, float | str]] = []

    probe_map = {
        "positive": ((2, "reserve_path", 2), (3, "atmosphere_control", 2)),
        "negative": ((5, "reserve_path", 3), (4, "atmosphere_control", 3)),
    }
    for branch, actuator_level, requested_reserve, initial in reserve_conditions(
        reserve_levels, pos_levels, neg_levels
    ):
        for valve_index, probe, target_index in probe_map[branch]:
            action = pulse_action(valve_index, args.pulse_physical)
            pulse, control = paired_counterfactual(
                sim, initial, prep, action, args.response_steps
            )
            valve = VALVE_NAMES[valve_index]
            base: dict[str, float | str] = {
                "branch": branch,
                "probe": probe,
                "valve": valve,
                "actuator_level_requested_kpa": actuator_level,
                "reserve_requested_kpa": requested_reserve,
                "pre_pos_reserve_kpa": float(pulse.pre_reserve[0]),
                "pre_neg_reserve_kpa": float(pulse.pre_reserve[1]),
                "pre_target_pressure_kpa": float(pulse.pre_pressure[target_index]),
                "pre_target_rate_kpa_s": float(pulse.pre_rate[target_index]),
            }
            for index, name in enumerate(PRESSURE_NAMES):
                base[f"pre_{name}_kpa"] = float(pulse.pre_pressure[index])
            for horizon in args.horizons:
                effect = counterfactual_effect(pulse, control, horizon)
                row = {
                    **base,
                    "horizon_steps": int(horizon),
                    "horizon_seconds": float(horizon * DT),
                    "target_counterfactual_effect_kpa": float(effect[target_index]),
                    "signed_authority_effect_kpa": float(
                        effect[target_index] if branch == "positive" else -effect[target_index]
                    ),
                    "integrated_probe_mdot_effect_kg": integrated_mdot_effect(
                        pulse, control, valve, horizon
                    ),
                }
                for index, name in enumerate(PRESSURE_NAMES):
                    row[f"effect_{name}_kpa"] = float(effect[index])
                summary_rows.append(row)
            run_id = f"{branch}_{actuator_level:g}_{requested_reserve:g}_{probe}"
            for step in range(args.response_steps + 1):
                trace: dict[str, float | str] = {
                    "run_id": run_id,
                    "branch": branch,
                    "probe": probe,
                    "step": step,
                    "time_seconds": step * DT,
                }
                for index, name in enumerate(PRESSURE_NAMES):
                    trace[f"pulse_{name}_kpa"] = float(pulse.response_pressure[step, index])
                    trace[f"control_{name}_kpa"] = float(control.response_pressure[step, index])
                trace_rows.append(trace)

    data = pd.DataFrame(summary_rows)
    traces = pd.DataFrame(trace_rows)
    data.to_csv(output_dir / "reserve_pulse_summary.csv", index=False)
    traces.to_csv(output_dir / "reserve_pulse_traces.csv", index=False)
    return data, traces


def regression_values(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if np.std(x) <= 1e-12:
        return {
            "slope_effect_per_reserve_kpa": 0.0,
            "intercept_kpa": float(np.mean(y)) if len(y) else 0.0,
            "r2": 0.0,
            "pearson_correlation": 0.0,
        }
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum(np.square(y - prediction)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-12 else 0.0
    return {
        "slope_effect_per_reserve_kpa": float(slope),
        "intercept_kpa": float(intercept),
        "r2": float(r2),
        "pearson_correlation": correlation,
    }


def linear_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (branch, probe, horizon), group in data.groupby(
        ["branch", "probe", "horizon_steps"]
    ):
        reserve_col = "pre_pos_reserve_kpa" if branch == "positive" else "pre_neg_reserve_kpa"
        x = group[reserve_col].to_numpy(dtype=float)
        y = group["signed_authority_effect_kpa"].to_numpy(dtype=float)
        rows.append(
            {
                "branch": branch,
                "probe": probe,
                "horizon_steps": int(horizon),
                "horizon_seconds": float(horizon * DT),
                "scope": "pooled",
                "actuator_level_requested_kpa": "all",
                "sample_count": int(len(group)),
                **regression_values(x, y),
            }
        )
        # Fixed-effect regression removes the mean of each requested actuator
        # pressure level before fitting reserve sensitivity.  This separates
        # reserve from absolute actuator pressure in the crossed grid.
        demeaned_x = group[reserve_col] - group.groupby(
            "actuator_level_requested_kpa"
        )[reserve_col].transform("mean")
        demeaned_y = group["signed_authority_effect_kpa"] - group.groupby(
            "actuator_level_requested_kpa"
        )["signed_authority_effect_kpa"].transform("mean")
        rows.append(
            {
                "branch": branch,
                "probe": probe,
                "horizon_steps": int(horizon),
                "horizon_seconds": float(horizon * DT),
                "scope": "within_level_fixed_effect",
                "actuator_level_requested_kpa": "demeaned",
                "sample_count": int(len(group)),
                **regression_values(
                    demeaned_x.to_numpy(dtype=float),
                    demeaned_y.to_numpy(dtype=float),
                ),
            }
        )
        for actuator_level, level_group in group.groupby(
            "actuator_level_requested_kpa"
        ):
            level_x = level_group[reserve_col].to_numpy(dtype=float)
            level_y = level_group["signed_authority_effect_kpa"].to_numpy(dtype=float)
            rows.append(
                {
                    "branch": branch,
                    "probe": probe,
                    "horizon_steps": int(horizon),
                    "horizon_seconds": float(horizon * DT),
                    "scope": "actuator_level",
                    "actuator_level_requested_kpa": float(actuator_level),
                    "sample_count": int(len(level_group)),
                    **regression_values(level_x, level_y),
                }
            )
    return pd.DataFrame(rows)


def plot_reserve(data: pd.DataFrame, output_dir: Path) -> None:
    horizons = sorted(data["horizon_steps"].unique())
    fig, axes = plt.subplots(2, len(horizons), figsize=(4.3 * len(horizons), 8), sharey="row")
    for row_index, branch in enumerate(("positive", "negative")):
        reserve_col = "pre_pos_reserve_kpa" if branch == "positive" else "pre_neg_reserve_kpa"
        for col_index, horizon in enumerate(horizons):
            axis = axes[row_index, col_index]
            selected = data[(data.branch == branch) & (data.horizon_steps == horizon)]
            for probe, group in selected.groupby("probe"):
                axis.scatter(
                    group[reserve_col],
                    group["signed_authority_effect_kpa"],
                    s=35,
                    alpha=0.8,
                    label=probe,
                )
            axis.set_title(f"{branch} | {horizon * DT:.2f} s")
            axis.set_xlabel("Measured reserve before pulse [kPa]")
            if col_index == 0:
                axis.set_ylabel("Signed pulse−control effect [kPa]")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)
    fig.suptitle("lib3 matched reserve intervention")
    fig.tight_layout()
    fig.savefig(output_dir / "reserve_pulse_effect.png", dpi=180)
    plt.close(fig)


def generate_history_actions(
    rng: np.random.Generator,
    branch: str,
    count: int,
    prep_steps: int,
) -> np.ndarray:
    if prep_steps < 6:
        raise ValueError("history prep requires at least 6 steps")
    actions = np.empty((count, prep_steps, 6), dtype=np.float64)
    split1 = prep_steps // 3
    split2 = 2 * prep_steps // 3
    for candidate in range(count):
        physical_segments = np.full((3, 6), PHYSICAL_LOW, dtype=np.float64)
        # Chamber actions and the active actuator inlet/outlet pair vary.
        physical_segments[:, 0:2] = rng.uniform(0.80, 0.96, size=(3, 2))
        if branch == "positive":
            physical_segments[:, 2:4] = rng.uniform(0.80, 0.99, size=(3, 2))
        else:
            physical_segments[:, 4:6] = rng.uniform(0.80, 0.99, size=(3, 2))
        # Keep the inactive branch at its lower command.
        segments = physical_to_raw(physical_segments)
        actions[candidate, :split1] = segments[0]
        actions[candidate, split1:split2] = segments[1]
        actions[candidate, split2:] = segments[2]
    return actions


def prepare_candidate(
    sim: PneuSim,
    candidate_id: int,
    branch: str,
    initial_pressure: tuple[float, float, float, float],
    actions: np.ndarray,
) -> Candidate:
    sim.set_init_press(*initial_pressure)
    pressure_rows: list[np.ndarray] = []
    for action in actions:
        obs, _ = sim.observe(action)
        pressure_rows.append(np.asarray(obs[1:5], dtype=np.float64).copy())
    pressure_array = np.asarray(pressure_rows)
    rate = (pressure_array[-1] - pressure_array[-2]) * FREQ_HZ
    return Candidate(
        candidate_id=candidate_id,
        branch=branch,
        actions=actions.copy(),
        pressure=pressure_array[-1],
        rate=rate,
        reserve=reserve_from_pressure(pressure_array[-1]),
        action_delta_norm=float(np.linalg.norm(actions[-1] - actions[-2])),
    )


def select_history_pairs(
    candidates: list[Candidate],
    branch: str,
    pair_count: int,
    pressure_max: float,
    reserve_max: float,
    min_rate: float,
    max_rate: float,
) -> list[tuple[Candidate, Candidate, float]]:
    target_index = 2 if branch == "positive" else 3
    proposals: list[tuple[float, int, int, float]] = []
    for i, first in enumerate(candidates):
        p1 = first.pressure
        r1 = first.rate[target_index]
        if abs(r1) < min_rate or abs(r1) > max_rate:
            continue
        for j in range(i + 1, len(candidates)):
            second = candidates[j]
            r2 = second.rate[target_index]
            if abs(r2) < min_rate or abs(r2) > max_rate or r1 * r2 >= 0.0:
                continue
            max_pressure_diff = float(np.max(np.abs(p1 - second.pressure)))
            if max_pressure_diff > pressure_max:
                continue
            max_reserve_diff = float(np.max(np.abs(first.reserve - second.reserve)))
            if max_reserve_diff > reserve_max:
                continue
            rate_separation = abs(r1 - r2)
            score = rate_separation / (0.25 + max_pressure_diff + 0.5 * max_reserve_diff)
            proposals.append((score, i, j, max_pressure_diff))
    proposals.sort(reverse=True)
    selected: list[tuple[Candidate, Candidate, float]] = []
    used: set[int] = set()
    for _, i, j, pressure_diff in proposals:
        first = candidates[i]
        second = candidates[j]
        if first.candidate_id in used or second.candidate_id in used:
            continue
        # Keep pair A as the rising target-pressure history for consistent plots.
        if first.rate[target_index] < second.rate[target_index]:
            first, second = second, first
        selected.append((first, second, pressure_diff))
        used.update((first.candidate_id, second.candidate_id))
        if len(selected) >= pair_count:
            break
    return selected


def run_history_matching(
    args: argparse.Namespace,
    sim: PneuSim,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(args.seed)
    initial = (160.0, 35.0, 125.0, 70.0)
    all_rows: list[dict[str, float | str]] = []
    pair_rows: list[dict[str, float | str]] = []
    trace_rows: list[dict[str, float | str]] = []
    action_archive: dict[str, np.ndarray] = {}

    for branch_index, branch in enumerate(("positive", "negative")):
        action_bank = generate_history_actions(
            rng,
            branch,
            args.history_candidates,
            args.history_prep_steps,
        )
        candidates: list[Candidate] = []
        for candidate_id, actions in enumerate(action_bank):
            candidate = prepare_candidate(
                sim,
                candidate_id,
                branch,
                initial,
                actions,
            )
            candidates.append(candidate)
            row: dict[str, float | str] = {
                "branch": branch,
                "candidate_id": candidate_id,
                "target_rate_kpa_s": float(candidate.rate[2 if branch == "positive" else 3]),
                "pos_reserve_kpa": float(candidate.reserve[0]),
                "neg_reserve_kpa": float(candidate.reserve[1]),
                "last_action_delta_norm": candidate.action_delta_norm,
            }
            for index, name in enumerate(PRESSURE_NAMES):
                row[f"{name}_kpa"] = float(candidate.pressure[index])
                row[f"rate_{name}_kpa_s"] = float(candidate.rate[index])
            all_rows.append(row)

        pairs = select_history_pairs(
            candidates,
            branch,
            args.history_pairs,
            args.pressure_match_max_kpa,
            args.reserve_match_max_kpa,
            args.minimum_opposite_rate_kpa_s,
            args.maximum_opposite_rate_kpa_s,
        )
        target_index = 2 if branch == "positive" else 3
        valve_index = 2 if branch == "positive" else 5
        action = pulse_action(valve_index, args.pulse_physical)
        for pair_id, (first, second, _) in enumerate(pairs):
            action_archive[f"{branch}_pair{pair_id}_rising"] = first.actions
            action_archive[f"{branch}_pair{pair_id}_falling"] = second.actions
            first_pulse, first_control = paired_counterfactual(
                sim, initial, first.actions, action, args.response_steps
            )
            second_pulse, second_control = paired_counterfactual(
                sim, initial, second.actions, action, args.response_steps
            )
            pressure_diff = first_pulse.pre_pressure - second_pulse.pre_pressure
            reserve_diff = first_pulse.pre_reserve - second_pulse.pre_reserve
            base: dict[str, float | str] = {
                "branch": branch,
                "pair_id": pair_id,
                "rising_candidate_id": first.candidate_id,
                "falling_candidate_id": second.candidate_id,
                "pre_pressure_rmse_kpa": float(np.sqrt(np.mean(np.square(pressure_diff)))),
                "pre_pressure_max_abs_diff_kpa": float(np.max(np.abs(pressure_diff))),
                "pre_reserve_max_abs_diff_kpa": float(np.max(np.abs(reserve_diff))),
                "rising_target_rate_kpa_s": float(first_pulse.pre_rate[target_index]),
                "falling_target_rate_kpa_s": float(second_pulse.pre_rate[target_index]),
                "target_rate_separation_kpa_s": float(
                    first_pulse.pre_rate[target_index] - second_pulse.pre_rate[target_index]
                ),
                "action_history_l2": float(np.linalg.norm(first.actions - second.actions)),
            }
            for horizon in args.horizons:
                first_effect = counterfactual_effect(first_pulse, first_control, horizon)
                second_effect = counterfactual_effect(second_pulse, second_control, horizon)
                pair_rows.append(
                    {
                        **base,
                        "horizon_steps": int(horizon),
                        "horizon_seconds": float(horizon * DT),
                        "rising_target_effect_kpa": float(first_effect[target_index]),
                        "falling_target_effect_kpa": float(second_effect[target_index]),
                        "history_conditioned_effect_difference_kpa": float(
                            first_effect[target_index] - second_effect[target_index]
                        ),
                    }
                )
            for step in range(args.response_steps + 1):
                trace_rows.append(
                    {
                        "branch": branch,
                        "pair_id": pair_id,
                        "step": step,
                        "time_seconds": step * DT,
                        "rising_pulse_target_kpa": float(first_pulse.response_pressure[step, target_index]),
                        "rising_control_target_kpa": float(first_control.response_pressure[step, target_index]),
                        "falling_pulse_target_kpa": float(second_pulse.response_pressure[step, target_index]),
                        "falling_control_target_kpa": float(second_control.response_pressure[step, target_index]),
                    }
                )
        if len(pairs) < args.history_pairs:
            print(
                f"[WARN] {branch}: selected {len(pairs)}/{args.history_pairs} history pairs "
                "under the requested matching tolerances"
            )

    candidates_df = pd.DataFrame(all_rows)
    pairs_df = pd.DataFrame(pair_rows)
    traces_df = pd.DataFrame(trace_rows)
    candidates_df.to_csv(output_dir / "history_candidates.csv", index=False)
    pairs_df.to_csv(output_dir / "history_pair_summary.csv", index=False)
    traces_df.to_csv(output_dir / "history_pair_traces.csv", index=False)
    np.savez_compressed(output_dir / "history_pair_actions.npz", **action_archive)
    return pairs_df, traces_df, action_archive


def plot_history_pairs(pairs: pd.DataFrame, output_dir: Path) -> None:
    if pairs.empty:
        return
    horizons = sorted(pairs["horizon_steps"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.3 * len(horizons), 4.5), sharey=True)
    if len(horizons) == 1:
        axes = np.asarray([axes])
    for axis, horizon in zip(axes, horizons):
        selected = pairs[pairs.horizon_steps == horizon]
        for branch, group in selected.groupby("branch"):
            axis.scatter(
                group["target_rate_separation_kpa_s"],
                group["history_conditioned_effect_difference_kpa"],
                s=40,
                label=branch,
            )
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_title(f"{horizon * DT:.2f} s")
        axis.set_xlabel("Matched-pair rate separation [kPa/s]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Pulse-effect difference [kPa]")
    fig.suptitle("lib3 history-matched pulse intervention")
    fig.tight_layout()
    fig.savefig(output_dir / "history_pair_effect.png", dpi=180)
    plt.close(fig)


def dataframe_records(data: pd.DataFrame) -> list[dict]:
    return json.loads(data.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.prep_steps < 2 or args.response_steps < max(args.horizons):
        raise ValueError("preparation/response steps are too short for requested horizons")
    if not PHYSICAL_LOW <= args.pulse_physical <= PHYSICAL_HIGH:
        raise ValueError("pulse physical command must lie in [0.8, 1.0]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = make_sim()

    reserve_data, _ = run_reserve_sweep(args, sim, args.output_dir)
    reserve_fit = linear_summary(reserve_data)
    reserve_fit.to_csv(args.output_dir / "reserve_pulse_regression.csv", index=False)
    plot_reserve(reserve_data, args.output_dir)

    history_pairs, _, action_archive = run_history_matching(args, sim, args.output_dir)
    plot_history_pairs(history_pairs, args.output_dir)

    metadata = {
        "purpose": "simulator matched-pulse mechanism diagnostic; not RL training",
        "conditions": {
            "freq_hz": FREQ_HZ,
            "sensor_delay_sec": 0.1,
            "physical_action_range": [PHYSICAL_LOW, PHYSICAL_HIGH],
            "baseline_physical_action": PHYSICAL_LOW,
            "pulse_physical_action": args.pulse_physical,
            "horizons_steps": args.horizons,
            "horizons_seconds": [step * DT for step in args.horizons],
        },
        "reserve_intervention": {
            "requested_reserve_levels_kpa": parse_csv_floats(args.reserve_levels).tolist(),
            "positive_actuator_levels_kpa": parse_csv_floats(args.pos_act_levels).tolist(),
            "negative_actuator_levels_kpa": parse_csv_floats(args.neg_act_levels).tolist(),
            "paired_control": "identical initialization/preparation, baseline response action",
            "negative_control": "atmosphere-connected actuator valve",
            "regression": dataframe_records(reserve_fit),
        },
        "history_intervention": {
            "random_seed": args.seed,
            "candidate_count_per_branch": args.history_candidates,
            "selected_action_sequence_count": len(action_archive),
            "requested_pairs_per_branch": args.history_pairs,
            "pressure_match_max_kpa": args.pressure_match_max_kpa,
            "reserve_match_max_kpa": args.reserve_match_max_kpa,
            "minimum_opposite_rate_kpa_s": args.minimum_opposite_rate_kpa_s,
            "maximum_opposite_rate_kpa_s": args.maximum_opposite_rate_kpa_s,
            "selected_pair_rows": int(len(history_pairs)),
        },
        "interpretation_limits": [
            "reserve initialization is a controlled simulator intervention, not a real preparation protocol",
            "history pairs are approximate matches, not identical full physical states",
            "hidden valve diagnostics are mechanism labels and must not enter the deployable actor",
            "real-system repetition is required for causal physical claims",
        ],
    }
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    print("\nReserve pulse regression")
    print(reserve_fit.to_string(index=False))
    print("\nHistory pair summary")
    if history_pairs.empty:
        print("no pairs selected")
    else:
        print(
            history_pairs.groupby(["branch", "horizon_steps"])[
                [
                    "pre_pressure_max_abs_diff_kpa",
                    "target_rate_separation_kpa_s",
                    "history_conditioned_effect_difference_kpa",
                ]
            ]
            .mean()
            .to_string()
        )
    print(f"saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
