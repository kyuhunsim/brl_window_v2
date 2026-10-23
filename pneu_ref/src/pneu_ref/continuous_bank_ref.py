"""Continuous bank playback with rate-limited minimum-jerk connecting segments."""
import bisect
import random
import numpy as np

from pneu_ref.base_ref import BaseRef
from pneu_ref.optimized_trajectory_ref import OptimizedTrajectoryBankRef


class ContinuousTrajectoryBankRef(BaseRef):
    def __init__(self, trajectory_npz, rate_limits, freq=50, seed=61098):
        super().__init__()
        self.trajectories = OptimizedTrajectoryBankRef(
            trajectory_npz=trajectory_npz, trajectory_index=0).trajectories
        self.rate_limits = np.array([[rate_limits[k]['up'], rate_limits[k]['down']]
                                     for k in ('act_pos', 'act_neg')], dtype=float)
        self.freq = float(freq)
        if not np.all(np.isfinite(self.rate_limits)) or np.any(self.rate_limits <= 0) or self.freq <= 0:
            raise ValueError('Continuous bank requires positive finite rate limits and frequency')
        self.seed = int(seed)
        self.time_reset()

    def time_reset(self):
        self.rng = random.Random(self.seed)
        rest = list(range(1, len(self.trajectories)))
        self.rng.shuffle(rest)
        self.pending = [0] + rest
        self.segments = []
        self.starts = []
        self.end_time = 0.0
        self.last_pressure = None
        self._append_member()

    def _append_member(self):
        if not self.pending:
            self.pending = list(range(len(self.trajectories)))
            self.rng.shuffle(self.pending)
        member = self.pending.pop(0)
        trajectory = self.trajectories[member]
        first = trajectory['pressure'][0]
        if self.last_pressure is not None:
            delta = first - self.last_pressure
            rates = np.where(delta >= 0, self.rate_limits[:, 0], self.rate_limits[:, 1])
            # max d/ds(10s^3 - 15s^4 + 6s^5) = 1.875.
            duration = np.ceil(1.875 * np.max(np.abs(delta) / rates) * self.freq) / self.freq
            if duration > 0:
                self.starts.append(self.end_time)
                self.segments.append(('bridge', self.end_time, duration,
                                      self.last_pressure.copy(), first.copy()))
                self.end_time += duration
        self.starts.append(self.end_time)
        duration = float(trajectory['time'][-1] - trajectory['time'][0])
        self.segments.append(('trajectory', self.end_time, duration, member))
        self.end_time += duration
        self.last_pressure = trajectory['pressure'][-1].copy()

    def get_goal(self, curr_time):
        t = max(float(curr_time), 0.0)
        if not np.isfinite(t):
            raise ValueError('Reference time must be finite')
        while t >= self.end_time:
            self._append_member()
        segment = self.segments[bisect.bisect_right(self.starts, t) - 1]
        kind, start, duration = segment[:3]
        if kind == 'bridge':
            s = np.clip((t - start) / duration, 0, 1)
            alpha = s**3 * (10 + s * (-15 + 6*s))
            result = segment[3] + alpha * (segment[4] - segment[3])
        else:
            trajectory = self.trajectories[segment[3]]
            local = t - start + trajectory['time'][0]
            result = [np.interp(local, trajectory['time'], trajectory['pressure'][:, i])
                      for i in range(2)]
        return tuple(result)
