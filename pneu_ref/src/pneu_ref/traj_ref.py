import numpy as np

class TrajRef():
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
    
    def get_goal(
        self,
        curr_time: float
    ) -> np.ndarray:
        while True:
            if curr_time < self.traj_time[0]:
                pos = self.traj_pos[0]
                neg = self.traj_neg[0]
                break
            elif len(self.traj_time) == 1:
                pos = self.traj_pos[0]
                neg = self.traj_neg[0]
                break
            elif curr_time >= self.traj_time[0] and curr_time < self.traj_time[1]:
                pos = self.traj_pos[0]
                neg = self.traj_neg[0]
                break
            else:
                self.traj_time = np.delete(self.traj_time, 0)
                self.traj_pos = np.delete(self.traj_pos, 0)
                self.traj_neg = np.delete(self.traj_neg, 0)
                continue
            
        return np.array([pos, neg])