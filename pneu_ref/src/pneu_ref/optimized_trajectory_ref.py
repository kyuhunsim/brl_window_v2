"""Reference bank for precomputed pressure trajectories.

The bank is agnostic to how a trajectory was designed.  A profile may come
from a simulator rollout, or from a pressure-space waypoint generator whose
targets were certified offline.  Only time and pressure are exposed through
this class; an optional oracle/action audit remains separate from training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np

from pneu_ref.base_ref import BaseRef


class OptimizedTrajectoryBankRef(BaseRef):
    """Sample pressure trajectories from a precomputed reference bank.

    ``time_reset`` selects a new trajectory for reset-style episodes.  For a
    carried rollout, the environment does not call ``time_reset`` at the
    horizon boundary, so the current trajectory continues in time.  When
    ``repeat_after_end`` is false, the last sample is held after the stored
    duration; it is therefore unsafe to use that profile beyond its recorded
    horizon.  When it is true, the bank members are concatenated at the
    boundary and the transition is reported in ``selection_log``.
    ``repeat_same=True`` instead wraps time within the selected trajectory,
    including look-ahead queries. ``trajectory_index`` pins selection across
    resets for reproducible evaluation. Wrapping does not smooth the endpoint
    back to the initial pressure.
    """

    def __init__(
        self,
        trajectories: list[Mapping[str, Any]] | None = None,
        trajectory_npz: str | None = None,
        seed: int = 61098,
        sample_mode: str = "random",
        repeat_after_end: bool = True,
        repeat_same: bool = False,
        trajectory_index: int | None = None,
    ) -> None:
        super().__init__()
        if trajectory_npz is not None:
            path = Path(trajectory_npz)
            if not path.is_file():
                raise FileNotFoundError(f"trajectory_npz does not exist: {path}")
            with np.load(path) as data:
                npz_time = np.asarray(data["time"], dtype=np.float64)
                npz_pressure = np.asarray(data["pressure"], dtype=np.float64)
            if npz_pressure.ndim == 2:
                npz_pressure = npz_pressure[None, ...]
            if npz_pressure.ndim != 3 or npz_pressure.shape[2] != 2:
                raise ValueError("trajectory_npz pressure must have shape (N, 2) or (B, N, 2)")
            if npz_time.ndim == 1:
                npz_time = np.repeat(npz_time[None, :], npz_pressure.shape[0], axis=0)
            trajectories = [
                {"id": index, "time": npz_time[index], "pressure": npz_pressure[index]}
                for index in range(npz_pressure.shape[0])
            ]
        if not trajectories:
            raise ValueError("optimized trajectory bank must contain trajectories or trajectory_npz")
        self.rng = random.Random(int(seed))
        self.sample_mode = str(sample_mode).lower()
        if self.sample_mode not in {"random", "coverage"}:
            raise ValueError("sample_mode must be 'random' or 'coverage'")
        self.repeat_after_end = bool(repeat_after_end)
        self.repeat_same = bool(repeat_same)
        self.trajectory_index = trajectory_index
        self.trajectories: list[dict[str, np.ndarray | float | int]] = []
        for index, item in enumerate(trajectories):
            times = np.asarray(item["time"], dtype=np.float64).reshape(-1)
            pressure = np.asarray(item["pressure"], dtype=np.float64)
            if pressure.ndim != 2 or pressure.shape[1] != 2:
                raise ValueError(f"trajectory {index} pressure must have shape (N, 2)")
            if len(times) != len(pressure) or len(times) < 2:
                raise ValueError(f"trajectory {index} time/pressure length mismatch")
            if not np.all(np.isfinite(times)) or not np.all(np.isfinite(pressure)):
                raise ValueError(f"trajectory {index} contains non-finite values")
            if np.any(np.diff(times) <= 0.0):
                raise ValueError(f"trajectory {index} time must be strictly increasing")
            if self.repeat_same and times[-1] <= 0:
                raise ValueError("Repeated trajectory duration must be positive")
            self.trajectories.append(
                {
                    "time": times,
                    "pressure": pressure,
                    "duration": float(times[-1]),
                    "id": int(item.get("id", index)),
                }
            )
        if trajectory_index is not None and not 0 <= trajectory_index < len(self.trajectories):
            raise ValueError("trajectory_index is outside the reference bank")
        self.visit_counts = np.zeros(len(self.trajectories), dtype=np.int64)
        self.selection_log: list[dict[str, float | int]] = []
        self._active_index = 0
        self._episode_index = 0
        self._time_origin = 0.0
        self.time_reset()

    def _select_index(self) -> int:
        if self.trajectory_index is not None:
            return self.trajectory_index
        if self.sample_mode == "coverage":
            minimum = int(np.min(self.visit_counts))
            candidates = np.flatnonzero(self.visit_counts == minimum)
            return int(self.rng.choice(candidates.tolist()))
        return int(self.rng.randrange(len(self.trajectories)))

    def time_reset(self) -> None:
        self._active_index = self._select_index()
        self.visit_counts[self._active_index] += 1
        self._time_origin = 0.0
        self._episode_index += 1
        self.selection_log.append(
            {
                "event": "reset",
                "episode": self._episode_index,
                "trajectory": int(self._active_index),
                "time": 0.0,
            }
        )

    def _advance_if_needed(self, time_value: float) -> float:
        local_time = float(time_value - self._time_origin)
        if self.repeat_after_end and self.repeat_same:
            # Stateless lookup: future-reference queries must not advance the
            # active trajectory or change subsequent current/past queries.
            duration = float(self.trajectories[self._active_index]["duration"])
            return max(local_time, 0.0) % duration
        if not self.repeat_after_end:
            return min(max(local_time, 0.0), float(self.trajectories[self._active_index]["duration"]))
        while local_time > float(self.trajectories[self._active_index]["duration"]):
            local_time -= float(self.trajectories[self._active_index]["duration"])
            self._active_index = (self._active_index + 1) % len(self.trajectories)
            self.visit_counts[self._active_index] += 1
            self.selection_log.append(
                {
                    "event": "carry_boundary",
                    "episode": self._episode_index,
                    "trajectory": int(self._active_index),
                    "time": float(time_value),
                }
            )
        return max(local_time, 0.0)

    def get_goal(self, curr_time: float) -> Tuple[float, float]:
        time_value = max(float(curr_time), 0.0)
        local_time = self._advance_if_needed(time_value)
        trajectory = self.trajectories[self._active_index]
        times = trajectory["time"]  # type: ignore[assignment]
        pressure = trajectory["pressure"]  # type: ignore[assignment]
        duration = float(trajectory["duration"])
        local_time = min(local_time, duration)
        return (
            float(np.interp(local_time, times, pressure[:, 0])),
            float(np.interp(local_time, times, pressure[:, 1])),
        )
