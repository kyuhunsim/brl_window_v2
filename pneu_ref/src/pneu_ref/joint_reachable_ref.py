"""Smooth random reference sampled from measured joint pressure points."""

from __future__ import annotations

import random
from typing import Iterable, Tuple

import numpy as np

from pneu_ref.base_ref import BaseRef


_MIN_JERK_PEAK_SLOPE = 1.875


class JointReachableRandomRef(BaseRef):
    """Generate smooth targets from a measured set of joint pressure pairs.

    Unlike a rectangular independent POS/NEG sampler, every waypoint is an
    observed pair from the plant command map.  The transition is a straight
    pressure-space segment with directional rate limits, so the target never
    invents a POS/NEG combination that was absent from characterization.
    """

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        initial_pos: float,
        initial_neg: float,
        pos_rate_up: float = 7.86,
        pos_rate_down: float = 16.0,
        neg_rate_up: float = 2.15,
        neg_rate_down: float = 4.47,
        hold_min: float = 1.5,
        hold_max: float = 4.0,
        min_transition: float = 2.0,
        max_frequency_hz: float | None = 0.175,
        initial_hold: float = 1.0,
        min_distance: float = 3.0,
        max_distance: float | None = 20.0,
        seed: int = 61098,
        selection_mode: str = "random",
        coverage_tie_margin: int = 0,
    ):
        super().__init__()
        point_array = np.asarray(list(points), dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[1] != 2 or len(point_array) < 2:
            raise ValueError("points must contain at least two [POS, NEG] pairs")
        if np.any(~np.isfinite(point_array)):
            raise ValueError("points must be finite")
        if any(value <= 0.0 for value in (pos_rate_up, pos_rate_down, neg_rate_up, neg_rate_down)):
            raise ValueError("all directional rates must be positive")
        if hold_min < 0.0 or hold_max < hold_min:
            raise ValueError("hold times must satisfy 0 <= hold_min <= hold_max")
        if min_transition <= 0.0:
            raise ValueError("min_transition must be positive")

        self.points = point_array
        self.initial = np.array([initial_pos, initial_neg], dtype=np.float64)
        self.rates = {
            "pos_up": float(pos_rate_up),
            "pos_down": float(pos_rate_down),
            "neg_up": float(neg_rate_up),
            "neg_down": float(neg_rate_down),
        }
        self.hold_bounds = (float(hold_min), float(hold_max))
        self.min_transition = float(min_transition)
        self.max_frequency_hz = None if max_frequency_hz is None else float(max_frequency_hz)
        self.initial_hold = max(float(initial_hold), 0.0)
        self.min_distance = max(float(min_distance), 0.0)
        self.max_distance = None if max_distance is None else max(float(max_distance), self.min_distance)
        self.rng = random.Random(seed)
        self.selection_mode = str(selection_mode).strip().lower().replace("-", "_")
        if self.selection_mode not in {
            "random",
            "coverage",
            "coverage_sweep",
            "coverage_sweep_neg",
        }:
            raise ValueError(
                "selection_mode must be one of: random, coverage, coverage_sweep, "
                "coverage_sweep_neg"
            )
        if coverage_tie_margin < 0:
            raise ValueError("coverage_tie_margin must be non-negative")
        self.coverage_tie_margin = int(coverage_tie_margin)
        # Counts intentionally survive time_reset().  A training run should
        # remember which measured joint points it has already visited across
        # episodes, including the current environment's two-reset lifecycle.
        self.visit_counts = np.zeros(len(self.points), dtype=np.int64)
        self._initialized = False
        self._segment_start = 0.0
        self._transition_start = 0.0
        self._transition_end = 0.0
        self._segment_end = 0.0
        self._start = self.initial.copy()
        self._target = self.initial.copy()

    def set_initial_state(self, pos: float, neg: float) -> None:
        self.initial = np.array([pos, neg], dtype=np.float64)
        self.time_reset()

    @staticmethod
    def _minimum_jerk(tau: float) -> float:
        tau = min(max(float(tau), 0.0), 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

    def _sample_target(self, previous: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.points - previous.reshape(1, 2), axis=1)
        candidate_indices = np.flatnonzero(distances >= self.min_distance)
        if self.max_distance is not None:
            local = np.flatnonzero(
                (distances >= self.min_distance) & (distances <= self.max_distance)
            )
            if len(local) > 0:
                candidate_indices = local
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(self.points))

        if self.selection_mode == "random":
            selected = int(candidate_indices[self.rng.randrange(len(candidate_indices))])
        else:
            # Coverage selection is still constrained by the local reachable
            # neighborhood above.  This prevents the sampler from jumping to
            # a remote pressure pair merely to improve coverage.
            counts = self.visit_counts[candidate_indices]
            min_count = int(np.min(counts))
            eligible = candidate_indices[counts <= min_count + self.coverage_tie_margin]
            if self.selection_mode == "coverage":
                selected = int(eligible[self.rng.randrange(len(eligible))])
            else:
                # coverage_sweep: among the least-visited local points, prefer
                # the farthest one.  This creates deliberate up/down sweeps
                # without leaving the measured joint-reachable set.
                if self.selection_mode == "coverage_sweep_neg":
                    # POS pressure spans a much larger numerical range than
                    # NEG pressure.  Euclidean distance would therefore hide
                    # local NEG motion.  This mode explicitly prioritizes the
                    # NEG-axis distance while still respecting joint
                    # reachability and coverage counts.
                    eligible_distances = np.abs(
                        self.points[eligible, 1] - previous.reshape(2)[1]
                    )
                else:
                    eligible_distances = distances[eligible]
                max_distance = float(np.max(eligible_distances))
                farthest = eligible[eligible_distances >= max_distance - 1e-9]
                selected = int(farthest[self.rng.randrange(len(farthest))])

        self.visit_counts[selected] += 1
        return self.points[selected].copy()

    def _rate(self, axis: str, delta: float) -> float:
        return self.rates[f"{axis}_{'up' if delta >= 0.0 else 'down'}"]

    def transition_duration(self, start: np.ndarray, target: np.ndarray) -> float:
        delta = np.asarray(target, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        duration = self.min_transition
        duration = max(
            duration,
            _MIN_JERK_PEAK_SLOPE * abs(float(delta[0])) / self._rate("pos", float(delta[0])),
            _MIN_JERK_PEAK_SLOPE * abs(float(delta[1])) / self._rate("neg", float(delta[1])),
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
            self._start_next_segment(self._segment_end, previous_target, initial=False)
        if time_value <= self._transition_start:
            goal = self._start
        elif time_value >= self._transition_end:
            goal = self._target
        else:
            tau = (time_value - self._transition_start) / (self._transition_end - self._transition_start)
            goal = self._start + self._minimum_jerk(tau) * (self._target - self._start)
        return float(goal[0]), float(goal[1])
