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

        # init_ref = self.ref.get_goal(1/ctrl_freq)
        init_ref = self.ref.get_goal(1/ctrl_freq)
        self.buf = np.c_[
            init_ref[0]*np.ones(self.num_ref, dtype=np.float64),
            init_ref[1]*np.ones(self.num_ref, dtype=np.float64),
        ].reshape(-1)

        self.step_time = (num_pred + 1)/ctrl_freq # (k+1)/f
    
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

        