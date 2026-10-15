"""Offline TD(lambda) discrepancy utilities.

The diagnostic follows the practical quantity used in the lambda-discrepancy
literature: compare two lambda-return targets evaluated on the same trace.
It is a representation diagnostic, not a proof that one closed-loop policy is
better.  For a fair comparison, traces should come from a common frozen policy
and common reference/mismatch condition whenever possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
    terminal_value: float = 0.0,
) -> np.ndarray:
    """Compute finite-trace TD(lambda) targets.

    ``rewards`` has length ``T``. ``values`` may contain ``V_0..V_T`` (length
    ``T+1``) or only ``V_0..V_{T-1}`` (length ``T``); in the latter case the
    supplied ``terminal_value`` is appended as ``V_T``.  The backward form is

    ``G[t] = r[t] + gamma * ((1-lambda)*V[t+1] + lambda*G[t+1])``.

    Thus lambda=0 is the one-step TD target and lambda approaching one uses a
    long-horizon return with the final bootstrap value.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if rewards.size == 0:
        return np.empty(0, dtype=np.float64)
    if values.size == rewards.size:
        values = np.r_[values, float(terminal_value)]
    if values.size != rewards.size + 1:
        raise ValueError(
            "values must have length len(rewards) or len(rewards)+1; "
            f"got {values.size} for {rewards.size} rewards"
        )
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError("rewards and values must be finite")
    gamma = float(gamma)
    lam = float(lam)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must lie in [0, 1], got {lam}")

    targets = np.empty(rewards.size, dtype=np.float64)
    continuation = float(values[-1])
    for index in range(rewards.size - 1, -1, -1):
        continuation = rewards[index] + gamma * (
            (1.0 - lam) * values[index + 1] + lam * continuation
        )
        targets[index] = continuation
    return targets


def discrepancy(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lambda_low: float = 0.0,
    lambda_high: float = 0.95,
    terminal_value: float = 0.0,
) -> dict[str, float | int]:
    """Return absolute/squared TD(lambda) discrepancy summary for one trace."""
    low = lambda_returns(rewards, values, gamma, lambda_low, terminal_value)
    high = lambda_returns(rewards, values, gamma, lambda_high, terminal_value)
    gap = high - low
    scale = float(np.mean(np.abs(0.5 * (high + low))))
    scale = max(scale, 1e-12)
    return {
        "n": int(gap.size),
        "lambda_low": float(lambda_low),
        "lambda_high": float(lambda_high),
        "mean_abs": float(np.mean(np.abs(gap))),
        "rmse": float(np.sqrt(np.mean(np.square(gap)))),
        "p95_abs": float(np.percentile(np.abs(gap), 95.0)),
        "max_abs": float(np.max(np.abs(gap))),
        "normalized_mean_abs": float(np.mean(np.abs(gap)) / scale),
        "return_scale": scale,
    }


def discrepancy_from_columns(
    rows: list[dict[str, Any]],
    reward_column: str = "reward",
    value_column: str = "value",
    **kwargs: Any,
) -> dict[str, float | int]:
    """Convenience wrapper for CSV-like dictionaries."""
    if not rows:
        raise ValueError("trace is empty")
    try:
        rewards = np.asarray([float(row[reward_column]) for row in rows])
        values = np.asarray([float(row[value_column]) for row in rows])
    except KeyError as exc:
        raise ValueError(
            f"trace must contain {reward_column!r} and {value_column!r} columns"
        ) from exc
    return discrepancy(rewards, values, **kwargs)

