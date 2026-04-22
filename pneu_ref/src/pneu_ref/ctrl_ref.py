from typing import List
import numpy as np

from pneu_ref.step_ref import StepRef
from pneu_ref.random_ref import RandomRef

class CtrlTraj:
    def __init__(
        self,
        traj_time: np.ndarray,
        traj_pos: np.ndarray,
        traj_neg: np.ndarray,
    ):
        assert len(traj_pos) == len(traj_neg), 'Trajectory lengthes of Pos and Neg are different!'
        assert len(traj_time) == len(traj_pos), 'Trajectory lengthes of Time and Press is different'
        self.traj_time = traj_time
        self.traj_pos = traj_pos
        self.traj_neg = traj_neg

        self.max_time = self.traj_time[-1]
    
    def get_ctrl(
        self,
        curr_time: float
    ) -> np.ndarray:
        ctrl_pos = 1
        ctrl_neg = 1
        while True:
            if curr_time < self.traj_time[0]:
                ctrl_pos = self.traj_pos[0]
                ctrl_neg = self.traj_neg[0]
                break
            elif len(self.traj_time) == 1:
                ctrl_pos = self.traj_pos[0]
                ctrl_neg = self.traj_neg[0]
                break
            elif curr_time >= self.traj_time[0] and curr_time < self.traj_time[1]:
                ctrl_pos = self.traj_pos[0]
                ctrl_neg = self.traj_neg[0]
                break
            else:
                self.traj_time = np.delete(self.traj_time, 0)
                self.traj_pos = np.delete(self.traj_pos, 0)
                self.traj_neg = np.delete(self.traj_neg, 0)
                continue
        return np.array([
            2*ctrl_pos - 1, 
            2*ctrl_neg - 1
        ])

class CtrlOnOff:
    def __init__(
        self,
        time_step: float,
        ctrl_pos: List[float] = [1, -1, 1, 1, 1, -1, 1],
        ctrl_neg: List[float] = [1, 1, 1, -1, 1, -1, 1]
    ):
        self.ctrl = StepRef(
            time_step = time_step,
            ref_pos = ctrl_pos,
            ref_neg = ctrl_neg
        )
        self.max_time = self.ctrl.max_time + 20
    
    def get_ctrl(
        self,
        curr_time: float
    ) -> np.ndarray:
        return np.array(self.ctrl.get_goal(curr_time))

class CtrlRamp:
    def __init__(
        self,
        time_step: float
    ):
        self.time_step = time_step
        self.ctrl_pos = [0, 1, 0, 0, 0, 1, 0]
        self.ctrl_neg = [0, 0, 0, 1, 0, 1, 0]

        self.max_time = self.time_step*(len(self.ctrl_pos)) + 20

    def get_ctrl(
        self,
        curr_time: float
    ) -> np.ndarray:
        pos_ctrl = 1
        neg_ctrl = 1
        
        for i, ctrl in enumerate(zip(self.ctrl_pos, self.ctrl_neg)):
            if curr_time < (i + 1)*self.time_step:
                ramp = (curr_time - i*self.time_step)/self.time_step
                pos_ctrl = 1 - 2*ramp*ctrl[0]
                neg_ctrl = 1 - 2*ramp*ctrl[1]
                break
        
        return np.array([pos_ctrl, neg_ctrl])

class CtrlRandom:
    def __init__(
        self,
        pos_max_ts = 10,    
        pos_max_amp = 0,
        pos_max_per = 3,
        pos_min_off = 900,
        pos_max_off = 1000,
        neg_max_ts = 10,    
        neg_max_amp = 0,
        neg_max_per = 3,
        neg_min_off = 900,
        neg_max_off = 1000,
    ):
        self.ctrl = RandomRef(
            pos_max_ts = pos_max_ts,    
            pos_max_amp = pos_max_amp,
            pos_max_per = pos_max_per,
            pos_min_off = pos_min_off,
            pos_max_off = pos_max_off,
            neg_max_ts = neg_max_ts,
            neg_max_amp = neg_max_amp,
            neg_max_per = neg_max_per,
            neg_min_off = neg_min_off,
            neg_max_off = neg_max_off
        )
        self.max_time = float('inf')
    
    def get_ctrl(
        self,
        curr_time: float
    ) -> np.ndarray:
        ctrl = self.ctrl.get_goal(curr_time)
        ctrl = 2*np.array(ctrl)/1000 - 1

        return ctrl
    
    def set_max_time(
        self,
        max_time: float
    ):
        self.max_time = max_time
        

if __name__ == '__main__':
    ctrl = CtrlRandom(
        time_step = 5
    )
    for i in range(2000):
        t = 0.01*i
        print(ctrl.get_ctrl(t))

        