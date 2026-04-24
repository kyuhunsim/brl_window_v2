import json
from typing import Sequence

import numpy as np


def make_stair_levels(start: float, peak: float, end: float, delta: float) -> list[float]:
    step = float(abs(delta))
    if step <= 0.0:
        raise ValueError("STAIR_DELTA must be > 0")

    lower = float(np.clip(start, 0.0, 1.0))
    upper = float(np.clip(peak, 0.0, 1.0))
    tail = float(np.clip(end, 0.0, 1.0))

    if upper < lower:
        lower, upper = upper, lower

    rising = [lower]
    value = lower
    while value + step < upper - 1e-12:
        value += step
        rising.append(round(value, 10))
    if abs(rising[-1] - upper) > 1e-12:
        rising.append(upper)

    falling = []
    value = upper
    while value - step > tail + 1e-12:
        value -= step
        falling.append(round(value, 10))
    if not falling or abs(falling[-1] - tail) > 1e-12:
        falling.append(tail)

    return [float(np.clip(x, 0.0, 1.0)) for x in rising + falling]


def stair_value(
    elapsed_s: float,
    *,
    levels: Sequence[float],
    hold_s: float,
    transition_s: float,
) -> float:
    t = float(max(0.0, elapsed_s))
    values = [float(v) for v in levels]
    hold = float(max(0.0, hold_s))

    if not values:
        return 0.0
    if len(values) == 1:
        return float(np.clip(values[0], 0.0, 1.0))

    if t < hold:
        return float(np.clip(values[0], 0.0, 1.0))
    t -= hold

    transition = float(max(0.0, transition_s))
    for idx in range(1, len(values)):
        prev_value = values[idx - 1]
        curr_value = values[idx]
        if transition > 0.0:
            if t < transition:
                alpha = t / transition
                return float(np.clip(prev_value + (curr_value - prev_value) * alpha, 0.0, 1.0))
            t -= transition
        if t < hold:
            return float(np.clip(curr_value, 0.0, 1.0))
        t -= hold

    return float(np.clip(values[-1], 0.0, 1.0))


def build_stair_ctrl(
    elapsed_s: float,
    *,
    n_ctrl: int,
    levels: Sequence[float],
    hold_s: float,
    transition_s: float,
    phase_offsets_s: Sequence[float],
) -> np.ndarray:
    if len(phase_offsets_s) != n_ctrl:
        raise ValueError(f"phase_offsets_s length must be {n_ctrl}, got {len(phase_offsets_s)}")

    ctrl = np.zeros(n_ctrl, dtype=np.float64)
    for idx in range(n_ctrl):
        ctrl[idx] = stair_value(
            elapsed_s + float(phase_offsets_s[idx]),
            levels=levels,
            hold_s=hold_s,
            transition_s=transition_s,
        )
    return np.clip(ctrl, 0.0, 1.0)


def read_flowrate_from_obs_json(obs_json_path: str) -> tuple[float, float, float, float, float, float]:
    try:
        with open(obs_json_path, "r", encoding="utf-8") as f:
            obs = json.load(f)
        return (
            float(obs.get("flowrate1", float("nan"))),
            float(obs.get("flowrate2", float("nan"))),
            float(obs.get("flowrate3", float("nan"))),
            float(obs.get("flowrate4", float("nan"))),
            float(obs.get("flowrate5", float("nan"))),
            float(obs.get("flowrate6", float("nan"))),
        )
    except Exception:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
