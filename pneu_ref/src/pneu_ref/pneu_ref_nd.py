from __future__ import annotations

from typing import Any

import numpy as np


class PneuRefND:
    """Reference trajectory buffer for refs with arbitrary fixed dimension."""

    def __init__(
        self,
        ref: Any,
        num_prev: int = 10,
        num_pred: int = 10,
        ctrl_freq: float = 50,
    ):
        self.ref = ref
        self.num_ref = num_prev + num_pred + 1
        self.ctrl_freq = ctrl_freq
        self.step_time = (num_pred + 1) / ctrl_freq
        self.dim_ref = 0
        self.reset()

    def _goal_array(self, curr_time: float) -> np.ndarray:
        goal = np.asarray(self.ref.get_goal(curr_time), dtype=np.float64).reshape(-1)
        if goal.size == 0:
            raise ValueError("reference goal must contain at least one value")
        if self.dim_ref and goal.size != self.dim_ref:
            raise ValueError(f"reference dimension changed from {self.dim_ref} to {goal.size}")
        return goal

    def reset(self) -> None:
        if hasattr(self.ref, "time_reset"):
            self.ref.time_reset()
        init_ref = self._goal_array(1 / self.ctrl_freq)
        self.dim_ref = int(init_ref.size)
        self.buf = np.tile(init_ref, (self.num_ref, 1)).reshape(-1)

    def get_ref(self, curr_time: float) -> np.ndarray:
        fut_time = curr_time + self.step_time
        ref = self._goal_array(fut_time)
        self.buf = np.r_[
            self.buf.copy().reshape(-1, self.dim_ref)[1:],
            ref.reshape(1, self.dim_ref),
        ].reshape(-1)
        return self.buf.copy()

    @property
    def goal_dim(self) -> int:
        return self.dim_ref * self.num_ref
