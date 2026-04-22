from typing import Dict
from collections import deque
import numpy as np

class PID():
    def __init__(
        self,
        Kp_pos: float = 1,
        Ki_pos: float = 1,
        Kd_pos: float = 0,
        Kp_neg: float = 1,
        Ki_neg: float = 1,
        Kd_neg: float = 0,
        freq: float = 50
    ):
        self.Kp = np.array([Kp_pos, Kp_neg], dtype=np.float32)
        self.Ki = np.array([Ki_pos, Ki_neg], dtype=np.float32)
        self.Kd = np.array([Kd_pos, Kd_neg], dtype=np.float32)
        
        self.sum = np.array([0, 0], dtype=np.float32)
        self.prev = np.array([0, 0], dtype=np.float32)

        self.dt = 1/freq

        self.is_anti_windup = False
    
    def get_action(
        self, 
        obs: np.ndarray, 
        ref: np.ndarray
    ) -> np.ndarray:
        err = ref - obs
        err = err*np.array([-1, 1], dtype=np.float32)
        self.sum += err*self.dt
        err_der = (err - self.prev)/(self.dt + 1e-8)
        
        action = \
            self.Kp*err \
            + self.Ki*self.sum \
            + self.Kd*err_der \
        
        self.prev = err

        return action
    
    def anti_windup(
        self,
        ctrl: np.ndarray,
        sat_ctrl: np.ndarray
    ) -> None:
        ctrl_err = ctrl - sat_ctrl
        self.sum -= self.Ka*ctrl_err

    def reset(self) -> None:
        self.sum = np.array([0, 0], dtype=np.float32)
        self.prev = np.array([0, 0], dtype=np.float32)
    
    def set_anti_windup(
        self,
        Ka: float
    ) -> None:
        self.is_anti_windup = True
        self.Ka = Ka

if __name__ == '__main__':
    pid = PID()
    
    action = pid.predict(
        np.array([1, 1]),
        np.array([0.99, 0.99])
    )
    pid.reset()
    print(pid.sum)