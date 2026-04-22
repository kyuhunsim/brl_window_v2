from typing import Optional, Tuple
from collections import deque

import numpy as np
import random
import math

from pneu_ref.base_ref import BaseRef
# from base_ref import BaseRef

class SineRef(BaseRef):
    def __init__(
        self, 
        pos_amp: float = 10,
        pos_per: float = 10,
        pos_off: int = 105,
        neg_amp: float = 5,
        neg_per: float = 10,
        neg_off: int = 15,
        iter: int = 1,
        buf_time: float = 20
    ):
        super(SineRef, self).__init__()

        self.pos_amp = pos_amp
        self.pos_per = pos_per
        self.pos_off = pos_off

        self.neg_amp = neg_amp
        self.neg_per = neg_per
        self.neg_off = neg_off

        self.max_time = np.lcm(pos_per, neg_per)*iter + buf_time
        self.buf_time = buf_time
        
    def get_goal(self, curr_time: float) -> Tuple[float, float]:
        if curr_time <= self.buf_time:
            return (self.pos_off, self.neg_off)
        
        curr_time -= self.buf_time
        pos_goal = self.get_sin_value(
            x = curr_time,
            amp = self.pos_amp,
            per = self.pos_per,
            off = self.pos_off
        )
        neg_goal = self.get_sin_value(
            x = curr_time,
            amp = self.neg_amp,
            per = self.neg_per,
            off = self.neg_off
        )
        return (
            pos_goal,
            neg_goal
        )
    
    def get_sin_value(
        self,
        x: float,
        amp: float,
        per: float,
        off: float
    ) -> np.ndarray:
        y = amp*np.sin(2*np.pi*x/per) + off

        return y

class DynamicOscillatorRef(BaseRef):
    def __init__(
        self,
        trans_time: float,
        pos_init_press: float,
        pos_final_press: float,
        pos_amp: float,
        pos_per: float,
        neg_init_press: float,
        neg_final_press: float,
        neg_amp: float,
        neg_per: float,
        buf_time: float = 20
    ):
        super(DynamicOscillatorRef, self).__init__()
        
        self.max_time = trans_time + buf_time
        self.init_press = np.array([pos_init_press, neg_init_press])
        self.final_press = np.array([pos_final_press, neg_final_press])
        self.amp = np.array([pos_amp, neg_amp])
        self.per = np.array([pos_per, neg_per])

        self.buf_time = buf_time
    
    def get_goal(
        self,
        curr_time: float
    ):
        if curr_time <= self.buf_time:
            # return self.init_press + self.amp*np.sin(2*np.pi/self.per*curr_time)
            return self.init_press
        else:
            curr_time -= self.buf_time
            return ((self.final_press - self.init_press)/self.max_time*curr_time + self.init_press) + self.amp*np.sin(2*np.pi/self.per*curr_time)

class CenterStepOscillationRef(BaseRef):
    def __init__(
        self,
        trans_time: float,
        pos_time_step: float,
        pos_center_step: float,
        pos_amp: float,
        pos_per: float,
        pos_init_press: float,
        neg_time_step: float,
        neg_center_step: float,
        neg_amp: float,
        neg_per: float,
        neg_init_press: float,
        buf_time: float = 10
    ):
        super(CenterStepOscillationRef, self).__init__()
        
        self.max_time = trans_time + buf_time
        self.buf_time = buf_time

        self.ts = np.array([pos_time_step, neg_time_step])
        self.cs = np.array([pos_center_step, neg_center_step])
        self.ci = np.array([pos_init_press, neg_init_press])
        self.A = np.array([pos_amp, neg_amp])
        self.T = np.array([pos_per, neg_per])
    
    def get_goal(
        self,
        curr_time: float, 
    ) -> np.ndarray:
        if curr_time <= self.buf_time:
            return self.ci
        else:
            t = curr_time - self.buf_time
            q = np.floor_divide(t, self.ts)
            r = np.mod(t, self.ts)

            return (self.cs*q + self.ci) + self.A*np.sin(2*np.pi/self.T*r)

if __name__ == '__main__':
    ref = SineRef()
    per = 10
    v = ref.get_sin_value(
        per/4,
        10,
        per,
        10
    )
    print(v)
    
