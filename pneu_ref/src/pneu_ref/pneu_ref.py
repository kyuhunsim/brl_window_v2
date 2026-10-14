from __future__ import annotations

from typing import Any, Tuple
from collections import deque

import numpy as np

class PneuRef():
    def __init__(
        self,
        ref: Any,
        num_prev: int = 10,
        num_pred: int = 10,
        ctrl_freq: float = 50
    ):
        self.ref = ref
        self.num_ref = num_prev + num_pred + 1 # reference traj dimension: R^{n+k+1}
        self.ctrl_freq = ctrl_freq

        self.reset()
        self.step_time = (num_pred + 1)/ctrl_freq # (k+1)/f

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        if initial_state is not None and hasattr(self.ref, "set_initial_state"):
            state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
            if state.size < 2:
                raise ValueError("pressure reference initial_state needs two values")
            self.ref.set_initial_state(float(state[0]), float(state[1]))
        if hasattr(self.ref, "time_reset"):
            self.ref.time_reset()
        init_ref = self.ref.get_goal(1/self.ctrl_freq)
        self.buf = np.c_[
            init_ref[0]*np.ones(self.num_ref, dtype=np.float64),
            init_ref[1]*np.ones(self.num_ref, dtype=np.float64),
        ].reshape(-1)
    
    def get_ref(
        self,
        curr_time: float
    ) -> np.ndarray:
        fut_time = curr_time + self.step_time
        ref = self.ref.get_goal(fut_time)

        self.buf = np.r_[
            self.buf.copy().reshape(-1,2)[1:],
            np.array(ref, dtype=np.float64).reshape(-1,2)
        ].reshape(-1)

        return self.buf.copy()

    @property
    def goal_dim(self) -> int:
        return 2*self.num_ref

if __name__ == '__main__':
    from random_ref import RandomRef
    ref = RandomRef()
    pneu_ref = PneuRef(ref)
    goal = pneu_ref.get_ref(0)
    print(goal)
    goal = pneu_ref.get_ref(10)
    print(goal)
    goal = pneu_ref.get_ref(20)
    print(goal)

        
