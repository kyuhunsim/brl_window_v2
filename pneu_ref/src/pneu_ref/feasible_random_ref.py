"""Physically rate-limited random pressure references.

The legacy :class:`RandomRef` changes sine parameters instantaneously.  Even
when each sine wave is slow, changing its offset or amplitude creates a
discontinuous pressure target.  This generator instead connects random
waypoints with a quintic minimum-jerk curve and selects the transition time
from directional plant pressure-rate limits.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np

from pneu_ref.base_ref import BaseRef


_MIN_JERK_PEAK_SLOPE = 1.875


class FeasibleRandomRef(BaseRef):
    """Generate bounded, smooth random pressure waypoints.

    All pressure values are absolute kPa and all rates are kPa/s.  A common
    transition duration is used for the two pressure channels so that the
    requested pressure-rate vector keeps a fixed direction during a segment.

    ``max_frequency_hz`` is a conservative plant-reference bandwidth limit.
    It imposes ``transition_time >= 1 / (2 f_max)`` in addition to the rate
    limit.  Set it to ``None`` to use only the rate and explicit minimum-time
    constraints.
    """

    def __init__(
        self,
        pos_min: float = 140.0,
        pos_max: float = 220.0,
        neg_min: float = 15.0,
        neg_max: float = 30.0,
        pos_rate_up: float = 12.0,
        pos_rate_down: float = 12.0,
        neg_rate_up: float = 5.0,
        neg_rate_down: float = 5.0,
        hold_min: float = 1.5,
        hold_max: float = 4.0,
        min_transition: float = 2.5,
        max_frequency_hz: float | None = 0.2,
        initial_pos: float = 101.325,
        initial_neg: float = 101.325,
        initial_hold: float = 1.0,
        min_pos_delta: float = 5.0,
        min_neg_delta: float = 2.0,
        seed: int = 61098,
    ):
        super().__init__()
        if not pos_min < pos_max:
            raise ValueError("pos_min must be less than pos_max")
        if not neg_min < neg_max:
            raise ValueError("neg_min must be less than neg_max")
        for name, value in {
            "pos_rate_up": pos_rate_up,
            "pos_rate_down": pos_rate_down,
            "neg_rate_up": neg_rate_up,
            "neg_rate_down": neg_rate_down,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if hold_min < 0.0 or hold_max < hold_min:
            raise ValueError("hold times must satisfy 0 <= hold_min <= hold_max")
        if min_transition <= 0.0:
            raise ValueError("min_transition must be positive")
        if max_frequency_hz is not None and max_frequency_hz <= 0.0:
            raise ValueError("max_frequency_hz must be positive or None")

        self.pos_bounds = (float(pos_min), float(pos_max))
        self.neg_bounds = (float(neg_min), float(neg_max))
        self.rates = {
            "pos_up": float(pos_rate_up),
            "pos_down": float(pos_rate_down),
            "neg_up": float(neg_rate_up),
            "neg_down": float(neg_rate_down),
        }
        self.hold_bounds = (float(hold_min), float(hold_max))
        self.min_transition = float(min_transition)
        self.max_frequency_hz = (
            None if max_frequency_hz is None else float(max_frequency_hz)
        )
        self.initial = np.array([initial_pos, initial_neg], dtype=np.float64)
        self.initial_hold = max(float(initial_hold), 0.0)
        self.min_delta = np.array(
            [max(float(min_pos_delta), 0.0), max(float(min_neg_delta), 0.0)],
            dtype=np.float64,
        )
        self.rng = random.Random(seed)

        self._initialized = False
        self._segment_start = 0.0
        self._transition_start = 0.0
        self._transition_end = 0.0
        self._segment_end = 0.0
        self._start = self.initial.copy()
        self._target = self.initial.copy()

    def set_initial_state(self, pos: float, neg: float) -> None:
        """Set the pressure state from which the next reset starts.

        A reference episode does not have to start at atmospheric pressure.
        The environment can call this at reset so a reset reference begins at
        the simulator's configured initial pressure.  For carried rollouts
        this method is not called at episode boundaries, so the reference
        remains continuous.
        """
        self.initial = np.asarray([pos, neg], dtype=np.float64)
        self._initialized = False
        self._segment_start = 0.0
        self._start = self.initial.copy()
        self._target = self.initial.copy()

    @staticmethod
    def _minimum_jerk(tau: float) -> float:
        tau = min(max(float(tau), 0.0), 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

    def _sample_axis(self, bounds: tuple[float, float], previous: float, minimum: float) -> float:
        low, high = bounds
        candidate = self.rng.uniform(low, high)
        for _ in range(31):
            if abs(candidate - previous) >= minimum:
                return candidate
            candidate = self.rng.uniform(low, high)
        # A very narrow configured range may make the requested minimum
        # impossible.  Return the farther endpoint instead of looping forever.
        return low if abs(low - previous) >= abs(high - previous) else high

    def _sample_target(self, previous: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self._sample_axis(self.pos_bounds, float(previous[0]), self.min_delta[0]),
                self._sample_axis(self.neg_bounds, float(previous[1]), self.min_delta[1]),
            ],
            dtype=np.float64,
        )

    def _rate_for_delta(self, axis: str, delta: float) -> float:
        direction = "up" if delta >= 0.0 else "down"
        return self.rates[f"{axis}_{direction}"]

    def transition_duration(
        self,
        start: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Return a duration that respects every configured rate limit."""
        delta = np.asarray(target, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        duration = self.min_transition
        duration = max(
            duration,
            _MIN_JERK_PEAK_SLOPE
            * abs(float(delta[0]))
            / self._rate_for_delta("pos", float(delta[0])),
            _MIN_JERK_PEAK_SLOPE
            * abs(float(delta[1]))
            / self._rate_for_delta("neg", float(delta[1])),
        )
        if self.max_frequency_hz is not None:
            duration = max(duration, 1.0 / (2.0 * self.max_frequency_hz))
        return float(duration)

    def _start_next_segment(self, start_time: float, start: np.ndarray, initial: bool) -> None:
        self._segment_start = float(start_time)
        self._start = np.asarray(start, dtype=np.float64).copy()
        self._target = self._sample_target(self._start)
        pre_hold = self.initial_hold if initial else 0.0
        duration = self.transition_duration(self._start, self._target)
        hold = self.rng.uniform(*self.hold_bounds)
        self._transition_start = self._segment_start + pre_hold
        self._transition_end = self._transition_start + duration
        self._segment_end = self._transition_end + hold
        self._initialized = True

    def time_reset(self) -> None:
        """Restart the clock while retaining the independent RNG stream."""
        self._initialized = False
        self._segment_start = 0.0
        self._start = self.initial.copy()
        self._target = self.initial.copy()

    def get_goal(self, curr_time: float) -> Tuple[float, float]:
        time_value = max(float(curr_time), 0.0)
        if not self._initialized:
            self._start_next_segment(0.0, self.initial, initial=True)

        while time_value >= self._segment_end:
            previous_target = self._target.copy()
            previous_end = self._segment_end
            self._start_next_segment(previous_end, previous_target, initial=False)

        if time_value <= self._transition_start:
            goal = self._start
        elif time_value >= self._transition_end:
            goal = self._target
        else:
            tau = (time_value - self._transition_start) / (
                self._transition_end - self._transition_start
            )
            blend = self._minimum_jerk(tau)
            goal = self._start + blend * (self._target - self._start)

        return float(goal[0]), float(goal[1])
