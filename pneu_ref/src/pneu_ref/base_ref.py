from typing import Tuple
import numpy as np

class BaseRef():
    def __init__(self):
        self.atm = 101.325
        self.max_time = float('inf')

    def get_goal(
        self,
        curr_time: float
    ) -> Tuple[float, float]:
        raise NotImplementedError
    
    
    

