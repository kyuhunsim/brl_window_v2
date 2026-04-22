from typing import Tuple, Optional
import numpy as np
from pneu_ref.base_ref import BaseRef
# from base_ref import BaseRef

class StepRef(BaseRef):
    def __init__(
        self,
        time_step: float,
        ref_pos: list,
        ref_neg: list,
        extra_time: Optional[float] = 0
    ):
        super(StepRef, self).__init__()
        assert len(ref_pos) == len(ref_neg), f'Lengthes of references are different (pos: {len(ref_pos)}, neg: {len(ref_neg)})'
        self.time_step = time_step
        self.ref_pos = ref_pos         
        self.ref_neg = ref_neg
        
        self.extra_time = time_step
        if extra_time is not None:
            self.extra_time = extra_time

        self.max_time = self.time_step*(len(self.ref_pos)) + self.extra_time

    def get_goal(self, curr_time: float) -> Tuple[float, float]:
        pos_goal = self.ref_pos[-1]
        neg_goal = self.ref_neg[-1]
        for i, goal in enumerate(zip(self.ref_pos, self.ref_neg)):
            if curr_time < (i + 1)*self.time_step:
                pos_goal = goal[0]
                neg_goal = goal[1]
                break
        
        return pos_goal, neg_goal
    
class StepCasesRef(StepRef):
    def __init__(
        self,
        time_step: float,
        ref_pos_max: float,
        ref_pos_min: float,
        ref_neg_max: float,
        ref_neg_min: float
    ):
        
        ref_pos_base = np.array([-1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1])
        ref_neg_base = np.array([1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1 ,1])
        
        ref_pos_scale = (ref_pos_max - ref_pos_min)/2
        ref_pos_bias = (ref_pos_max + ref_pos_min)/2
        ref_neg_scale = (ref_neg_max - ref_neg_min)/2
        ref_neg_bias = (ref_neg_max + ref_neg_min)/2

        ref_pos = ref_pos_scale*ref_pos_base + ref_pos_bias
        ref_neg = ref_neg_scale*ref_neg_base + ref_neg_bias

        super(StepCasesRef, self).__init__(
            time_step = time_step,
            ref_pos = ref_pos,
            ref_neg = ref_neg   
        )
    

        
