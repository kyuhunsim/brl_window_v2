#!/usr/bin/env python3
"""Fixed, physical-step metrics for the lib4 follow-up plan.

The protocol deliberately separates transition tracking from plateau
stability.  It never searches for the lag or derivative threshold that makes
a particular checkpoint look best; both must be fixed in a YAML file before
evaluation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = yaml.safe_load(raw)
    if protocol.get("schema") != "lib4_evaluation_protocol_v1":
        raise ValueError(f"unsupported lib4 evaluation protocol: {path}")
    return protocol, hashlib.sha256(raw).hexdigest()


def rmse(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan")


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def direction_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    dt: float,
    min_reference_rate: float,
    lag_steps: int,
) -> dict[str, float]:
    """Compare signed rates after a fixed causal actual-response lag."""
    reference_rate = np.diff(np.asarray(reference, dtype=np.float64)) / dt
    actual_rate = np.diff(np.asarray(actual, dtype=np.float64)) / dt
    lag = max(int(lag_steps), 0)
    if lag:
        if len(reference_rate) <= lag:
            return {"agreement": float("nan"), "correlation": float("nan"), "coverage": 0.0}
        reference_rate = reference_rate[:-lag]
        actual_rate = actual_rate[lag:]
    moving = np.abs(reference_rate) >= float(min_reference_rate)
    if not bool(np.any(moving)):
        return {"agreement": float("nan"), "correlation": float("nan"), "coverage": 0.0}
    ref = reference_rate[moving]
    act = actual_rate[moving]
    correlation = (
        float(np.corrcoef(ref, act)[0, 1])
        if len(ref) >= 2 and np.std(ref) > 0.0 and np.std(act) > 0.0
        else float("nan")
    )
    return {
        "agreement": float(np.mean(np.sign(ref) == np.sign(act))),
        "correlation": correlation,
        "coverage": float(np.mean(moving)),
    }


def _runs(mask: np.ndarray, minimum_length: int) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, enabled in enumerate(np.r_[mask, False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            if index - start >= minimum_length:
                yield start, index
            start = None


def segmented_tracking_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    dt: float,
    transition_rate_min: float,
    plateau_rate_max: float,
    minimum_plateau_samples: int,
    settling_tolerance: float,
    settling_hold_samples: int,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    error = reference - actual
    reference_rate = np.abs(np.diff(reference, prepend=reference[0])) / dt
    transition = reference_rate >= float(transition_rate_min)
    plateau = reference_rate <= float(plateau_rate_max)
    plateau_runs = list(_runs(plateau, max(int(minimum_plateau_samples), 1)))

    residual_parts: list[np.ndarray] = []
    plateau_p2p: list[float] = []
    plateau_rate_parts: list[np.ndarray] = []
    for start, end in plateau_runs:
        segment_error = error[start:end]
        residual_parts.append(segment_error - np.mean(segment_error))
        plateau_p2p.append(float(np.ptp(segment_error)))
        if end - start >= 2:
            plateau_rate_parts.append(np.diff(actual[start:end]) / dt)

    residual = np.concatenate(residual_parts) if residual_parts else np.empty(0)
    plateau_actual_rate = (
        np.concatenate(plateau_rate_parts) if plateau_rate_parts else np.empty(0)
    )

    settling_times: list[float] = []
    hold = max(int(settling_hold_samples), 1)
    transition_runs = list(_runs(transition, 1))
    for _, transition_end in transition_runs:
        next_transition = next(
            (start for start, _ in transition_runs if start >= transition_end),
            len(error),
        )
        within = np.abs(error[transition_end:next_transition]) <= float(settling_tolerance)
        for offset in range(0, max(len(within) - hold + 1, 0)):
            if bool(np.all(within[offset:offset + hold])):
                settling_times.append(offset * dt)
                break

    return {
        "transition_rmse": rmse(error[transition]),
        "transition_fraction": float(np.mean(transition)),
        "plateau_rmse": rmse(error[plateau]),
        "plateau_bias": _safe_mean(error[plateau]),
        "plateau_fraction": float(np.mean(plateau)),
        "plateau_segment_count": float(len(plateau_runs)),
        "plateau_oscillation_std": float(np.std(residual)) if residual.size else float("nan"),
        "plateau_error_p2p_max": max(plateau_p2p, default=float("nan")),
        "plateau_actual_rate_rms": rmse(plateau_actual_rate),
        "settling_time_mean_sec": _safe_mean(np.asarray(settling_times, dtype=np.float64)),
        "settling_time_max_sec": max(settling_times, default=float("nan")),
        "settled_transition_fraction": (
            float(len(settling_times) / len(transition_runs)) if transition_runs else float("nan")
        ),
    }


def summarize_validation(
    rows: list[dict[str, float]],
    *,
    member: int,
    action_chunk_steps: int,
    protocol: dict[str, Any],
) -> dict[str, float | int | bool]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    dt = float(protocol["sample_period_sec"])
    channels = protocol["channels"]
    direction_cfg = protocol["direction"]
    segment_cfg = protocol["segmentation"]
    safety_cfg = protocol["safety"]
    gates = protocol["provisional_gates"]

    actual = {name: values(cfg["actual"]) for name, cfg in channels.items()}
    reference = {name: values(cfg["reference"]) for name, cfg in channels.items()}
    errors = {name: reference[name] - actual[name] for name in channels}
    controls = np.stack([values(key) for key in protocol["control_columns"]], axis=1)
    delta = np.abs(np.diff(controls, axis=0))
    delta2 = np.abs(np.diff(controls, n=2, axis=0))

    summary: dict[str, float | int | bool] = {
        "member": int(member),
        "actual_steps": int(len(rows)),
        "duration_sec": float(values("time")[-1] - values("time")[0]),
    }
    for name, cfg in channels.items():
        summary[f"{name}_rmse"] = rmse(errors[name])
        summary[f"{name}_mae"] = float(np.mean(np.abs(errors[name])))
        summary[f"{name}_bias"] = float(np.mean(errors[name]))
        direction = direction_metrics(
            reference[name],
            actual[name],
            dt=dt,
            min_reference_rate=float(direction_cfg[cfg["kind"]]["min_reference_rate"]),
            lag_steps=int(direction_cfg[cfg["kind"]]["lag_steps"]),
        )
        for metric, value in direction.items():
            summary[f"{name}_direction_{metric}"] = value
        segmented = segmented_tracking_metrics(
            reference[name],
            actual[name],
            dt=dt,
            transition_rate_min=float(segment_cfg[cfg["kind"]]["transition_rate_min"]),
            plateau_rate_max=float(segment_cfg[cfg["kind"]]["plateau_rate_max"]),
            minimum_plateau_samples=int(segment_cfg["minimum_plateau_samples"]),
            settling_tolerance=float(segment_cfg[cfg["kind"]]["settling_tolerance"]),
            settling_hold_samples=int(segment_cfg["settling_hold_samples"]),
        )
        for metric, value in segmented.items():
            summary[f"{name}_{metric}"] = value

    pos_reserve = values("ch_pos") - values("act_pos")
    neg_reserve = values("act_neg") - values("ch_neg")
    summary.update(
        pos_reserve_min=float(np.min(pos_reserve)),
        neg_reserve_min=float(np.min(neg_reserve)),
        pos_reserve_violation_fraction=float(
            np.mean(pos_reserve < float(safety_cfg["pos_reserve_min_kpa"]))
        ),
        neg_reserve_violation_fraction=float(
            np.mean(neg_reserve < float(safety_cfg["neg_reserve_min_kpa"]))
        ),
        ch_pos_max=float(np.max(values("ch_pos"))),
        ch_neg_min=float(np.min(values("ch_neg"))),
        disp_min=float(np.min(values("disp"))),
        disp_max=float(np.max(values("disp"))),
        disp_span=float(np.ptp(values("disp"))),
        disp_ref_span=float(np.ptp(values("ref_disp"))),
        mean_abs_delta_u=float(np.mean(delta)) if delta.size else 0.0,
        max_abs_delta_u=float(np.max(delta)) if delta.size else 0.0,
        mean_abs_delta2_u=float(np.mean(delta2)) if delta2.size else 0.0,
        max_abs_delta2_u=float(np.max(delta2)) if delta2.size else 0.0,
        action_saturation_ratio=float(
            np.mean(
                (controls <= float(safety_cfg["control_saturation_low"]))
                | (controls >= float(safety_cfg["control_saturation_high"]))
            )
        ),
    )

    if action_chunk_steps > 1 and "policy_step" in rows[0]:
        policy_step = values("policy_step").astype(np.int64)
        within = policy_step[1:] == policy_step[:-1]
        boundary = ~within
        summary["within_chunk_mean_abs_delta_u"] = (
            float(np.mean(delta[within])) if bool(np.any(within)) else float("nan")
        )
        summary["boundary_mean_abs_delta_u"] = (
            float(np.mean(delta[boundary])) if bool(np.any(boundary)) else float("nan")
        )
        threshold = float(gates["large_action_delta"])
        summary["within_chunk_large_delta_fraction"] = (
            float(np.mean(delta[within] > threshold)) if bool(np.any(within)) else float("nan")
        )
        summary["boundary_large_delta_fraction"] = (
            float(np.mean(delta[boundary] > threshold)) if bool(np.any(boundary)) else float("nan")
        )

    gate_values = {
        "pos_rmse": summary["pos_rmse"] <= float(gates["pos_rmse_max_kpa"]),
        "neg_rmse": summary["neg_rmse"] <= float(gates["neg_rmse_max_kpa"]),
        "disp_rmse": summary["disp_rmse"] <= float(gates["disp_rmse_max_mm"]),
        "pos_direction": summary["pos_direction_agreement"] >= float(gates["direction_min"]),
        "neg_direction": summary["neg_direction_agreement"] >= float(gates["direction_min"]),
        "disp_direction": summary["disp_direction_agreement"] >= float(gates["direction_min"]),
        "pos_reserve": summary["pos_reserve_min"] >= float(safety_cfg["pos_reserve_min_kpa"]),
        "neg_reserve": summary["neg_reserve_min"] >= float(safety_cfg["neg_reserve_min_kpa"]),
        "action_variation": summary["mean_abs_delta_u"] <= float(gates["mean_abs_delta_u_max"]),
        "pos_plateau": summary["pos_plateau_oscillation_std"] <= float(gates["pressure_plateau_std_max_kpa"]),
        "neg_plateau": summary["neg_plateau_oscillation_std"] <= float(gates["pressure_plateau_std_max_kpa"]),
        "disp_plateau": summary["disp_plateau_oscillation_std"] <= float(gates["disp_plateau_std_max_mm"]),
    }
    for name, passed in gate_values.items():
        summary[f"gate_{name}"] = bool(passed)
    summary["gate_all"] = bool(all(gate_values.values()))
    return summary

