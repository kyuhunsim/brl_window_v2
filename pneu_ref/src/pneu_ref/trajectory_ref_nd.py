"""Reference bank for fixed-dimensional trajectories, including lib4 3D goals.

Unlike the legacy pressure-only trajectory bank, this class preserves every
reference channel stored in the NPZ.  It is used for lib4's jointly feasible
``[act_pos, act_neg, displacement]`` trajectories.  Oracle actions can live
in the same NPZ as diagnostics, but are deliberately never exposed here.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from pneu_ref.base_ref import BaseRef


class TrajectoryBankRefND(BaseRef):
    """Sample and replay precomputed fixed-dimensional reference trajectories."""

    def __init__(
        self,
        trajectory_npz: str | Path,
        reference_key: str = "reference",
        seed: int = 61098,
        sample_mode: str = "coverage",
        repeat_after_end: bool = False,
    ) -> None:
        super().__init__()
        path = Path(trajectory_npz)
        if not path.is_file():
            raise FileNotFoundError(f"trajectory_npz does not exist: {path}")
        with np.load(path) as data:
            if "time" not in data or reference_key not in data:
                raise ValueError(
                    f"{path} must contain 'time' and {reference_key!r} arrays"
                )
            time = np.asarray(data["time"], dtype=np.float64)
            reference = np.asarray(data[reference_key], dtype=np.float64)
        if reference.ndim == 2:
            reference = reference[None, ...]
        if reference.ndim != 3 or reference.shape[0] == 0 or reference.shape[1] < 2:
            raise ValueError(
                "reference must have shape (steps, channels) or "
                "(bank, steps, channels)"
            )
        if time.ndim == 1:
            time = np.repeat(time[None, :], reference.shape[0], axis=0)
        if time.shape != reference.shape[:2]:
            raise ValueError(
                f"time shape {time.shape} must match reference bank/steps "
                f"{reference.shape[:2]}"
            )
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(reference)):
            raise ValueError("trajectory bank contains non-finite values")
        if np.any(np.diff(time, axis=1) <= 0.0):
            raise ValueError("trajectory time must be strictly increasing")

        self.path = path
        self.time = time
        self.reference = reference
        self.reference_key = str(reference_key)
        self.dim_ref = int(reference.shape[2])
        self.rng = random.Random(int(seed))
        self.sample_mode = str(sample_mode).lower()
        if self.sample_mode not in {"random", "coverage"}:
            raise ValueError("sample_mode must be 'random' or 'coverage'")
        self.repeat_after_end = bool(repeat_after_end)
        self.visit_counts = np.zeros(reference.shape[0], dtype=np.int64)
        self.selection_log: list[dict[str, float | int | str]] = []
        self._active_index = 0
        self._time_origin = 0.0
        self._episode_index = 0
        self.time_reset()

    def _select_index(self) -> int:
        if self.sample_mode == "coverage":
            candidates = np.flatnonzero(self.visit_counts == np.min(self.visit_counts))
            return int(self.rng.choice(candidates.tolist()))
        return int(self.rng.randrange(len(self.reference)))

    def time_reset(self) -> None:
        self._active_index = self._select_index()
        self.visit_counts[self._active_index] += 1
        self._time_origin = 0.0
        self._episode_index += 1
        self.selection_log.append(
            {
                "event": "reset",
                "episode": self._episode_index,
                "trajectory": self._active_index,
                "time": 0.0,
            }
        )

    def _local_time(self, curr_time: float) -> float:
        local_time = max(float(curr_time) - self._time_origin, 0.0)
        duration = float(self.time[self._active_index, -1])
        if not self.repeat_after_end:
            return min(local_time, duration)
        while local_time > duration:
            local_time -= duration
            self._active_index = (self._active_index + 1) % len(self.reference)
            self.visit_counts[self._active_index] += 1
            self.selection_log.append(
                {
                    "event": "carry_boundary",
                    "episode": self._episode_index,
                    "trajectory": self._active_index,
                    "time": float(curr_time),
                }
            )
            duration = float(self.time[self._active_index, -1])
        return local_time

    def get_goal(self, curr_time: float) -> tuple[float, ...]:
        local_time = self._local_time(curr_time)
        time = self.time[self._active_index]
        values = self.reference[self._active_index]
        return tuple(
            float(np.interp(local_time, time, values[:, channel]))
            for channel in range(self.dim_ref)
        )
