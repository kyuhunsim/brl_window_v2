from typing import Optional
from collections import deque

import numpy as np
import random
import math

from pneu_ref.base_ref import BaseRef
# from base_ref import BaseRef

class RandomRef(BaseRef):
    def __init__(
        self, 
        pos_max_ts: int = 5,
        pos_max_amp: float = 10,
        pos_max_per: float = 10,
        pos_min_off: int = 105,
        pos_max_off: int = 245,
        neg_max_ts: float = 5,
        neg_max_amp: float = 5,
        neg_max_per: float = 10,
        neg_min_off: int = 15,
        neg_max_off: int = 90,
        # seed: int = 113000,
        seed: int = 61098,
    ):
        super(RandomRef, self).__init__()
        
        random.seed(seed)

        self.pos_start_time = 0
        self.pos_time_step = 0
        self.pos_amplitude = 0
        self.pos_period = 1
        self.pos_offset = 0

        self.neg_start_time = 0
        self.neg_time_step = 0
        self.neg_amplitude = 0
        self.neg_period = 1
        self.neg_offset = 0

        self.atm = 101.325

        self.pos_max_ts = pos_max_ts
        self.pos_max_amp = pos_max_amp 
        self.pos_max_per = pos_max_per 
        self.pos_min_off = pos_min_off 
        self.pos_max_off = pos_max_off 

        self.neg_max_ts = neg_max_ts 
        self.neg_max_amp = neg_max_amp 
        self.neg_max_per = neg_max_per 
        self.neg_min_off = neg_min_off 
        self.neg_max_off = neg_max_off 

        
    def get_goal(self, curr_time):
        pos_goal = self.pos_random_goal(
            curr_time,
            max_time_step = self.pos_max_ts,
            max_amplitude = self.pos_max_amp,
            max_period = self.pos_max_per,
            min_offset = self.pos_min_off,
            max_offset = self.pos_max_off
        )
        
        neg_goal = self.neg_random_goal(
            curr_time,
            max_time_step = self.neg_max_ts,
            max_amplitude = self.neg_max_amp,
            max_period = self.neg_max_per,
            min_offset = self.neg_min_off,
            max_offset = self.neg_max_off
        )

        # goal = np.array([pos_goal + self.atm, neg_goal + self.atm], dtype=np.float64)
        
        return (
            pos_goal,
            neg_goal
        )

    def time_reset(self):
        self.pos_start_time = 0
        self.neg_start_time = 0
    
    def pos_random_goal(
        self, 
        T,                 
        max_time_step,
        max_amplitude,
        max_period,
        min_offset,
        max_offset
    ):
        '''
        y = A sin(2*pi*t/T) + B
        '''
        dt = T - self.pos_start_time
        if dt > self.pos_time_step:
            self.pos_start_time = T 
            self.pos_time_step = random.randrange(1, max_time_step + 1)
            self.pos_amplitude = random.randrange(-max_amplitude, max_amplitude + 1)
            self.pos_period = 0.1*random.randrange(20, 10*max_period + 1)
            self.pos_offset = random.randrange(min_offset, max_offset + 1)
            
        goal = self.pos_amplitude*math.sin(2*math.pi*dt/self.pos_period) + self.pos_offset
        return goal

    def neg_random_goal(
        self, 
        T,                 
        max_time_step,
        max_amplitude,
        max_period,
        min_offset,
        max_offset
    ):
        '''
        y = A sin(2*pi*t/T) + B
        '''
        dt = T - self.neg_start_time
        if dt > self.neg_time_step:
            self.neg_start_time = T 
            self.neg_time_step = random.randrange(1, max_time_step + 1)
            self.neg_amplitude = random.randrange(-max_amplitude, max_amplitude + 1)
            self.neg_period = 0.1*random.randrange(20, 10*max_period + 1)
            self.neg_offset = random.randrange(min_offset, max_offset + 1)
            
        goal = self.neg_amplitude*math.sin(2*math.pi*dt/self.neg_period) + self.neg_offset
        return goal
    
 