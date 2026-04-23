from typing import Dict

import numpy as np
import time
import threading

import json

from pneu_utils.utils import get_pkg_path, color
from pneu_env.pid import PID


class PneuReal():
    def __init__(
        self,
        freq: float = 50,
        scale: bool = False
    ):
        self.freq = freq
        self.sen_period = 1/freq
        
        self.curr_time = 0.0
        self.sen_pos = 101.325
        self.sen_neg = 101.325
        self.ref_pos = 101.325
        self.ref_neg = 101.325
        self.ctrl_pos = 1.0
        self.ctrl_neg = 1.0
        
        self.flag_time = time.time()
        self.start_time = time.time()
        
        self.scale = scale

        self.labview_path = f"{get_pkg_path('pneu_env')}/src/pneu_env/tcpip"

        self.stop_flag = threading.Event()

        self.is_pid = False
        self.is_anti_windup = False
        self.obs = np.array([101.325, 101.325], dtype=np.float32)

    def write_ctrl_file(self) -> None:
        data = dict(
            time = float(time.time() - self.start_time),
            sen_pos = float(self.sen_pos),
            sen_neg = float(self.sen_neg),
            ref_pos = float(self.ref_pos),
            ref_neg = float(self.ref_neg),
            ctrl_pos = float(self.ctrl_pos),
            ctrl_neg = float(self.ctrl_neg)
        )
        with open(f'{self.labview_path}/ctrl.json', 'w') as f:
            json.dump(data, f)

    def read_obs_file(self) -> None:
        while not self.stop_flag.is_set():
            try:
                with open(f'{self.labview_path}/obs.json', 'r') as f:
                    obs = json.load(f)
                    self.msg = dict()
                    for i, v in enumerate(obs.values()):
                        self.msg[f"msg{i}"] = v
                    # self.curr_time = obs['time']
                    self.curr_time = time.time() - self.start_time
                    self.sen_pos = obs['sen_pos']
                    self.sen_neg = obs['sen_neg']
                break
            except:
                continue
    
    def wait(
        self,
        margin: float = 0.004
    ) -> None:
        curr_flag_time = time.time()
        time.sleep(max(self.sen_period - curr_flag_time + self.flag_time - margin, 0))
        self.flag_time = time.time()

    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        if self.is_pid:
            err = self.pid.get_action(self.obs, goal)
            ctrl += err
            if self.is_anti_windup:
                original_ctrl = ctrl
    
        ctrl = np.clip(ctrl, -1, 1)

        if self.is_anti_windup:
            sat_ctrl = ctrl
            self.pid.anti_windup(
                ctrl = original_ctrl,
                sat_ctrl = sat_ctrl
            )
        
        if self.scale:
            ctrl = 0.3*0.5*(ctrl + 1) + 0.7
        else:
            ctrl = 0.5*ctrl + 0.5

        self.ref_pos = goal[0]
        self.ref_neg = goal[1]
        self.ctrl_pos = ctrl[0]
        self.ctrl_neg = ctrl[1]
        
        self.write_ctrl_file()
        self.wait()
        self.read_obs_file()
        
        next_obs = np.array([
            self.curr_time,
            self.sen_pos,
            self.sen_neg
        ])
        self.obs = next_obs[1:3]

        Observation_info = dict(
            curr_time = self.curr_time,
            sen_pos = self.sen_pos,
            sen_neg = self.sen_neg,
            ref_pos = self.ref_pos,
            ref_neg = self.ref_neg,
            ctrl_pos = self.ctrl_pos,
            ctrl_neg = self.ctrl_neg
        )

        info = {}
        info['obs_w/o_noise'] = np.array([
            Observation_info['curr_time'],
            Observation_info['sen_pos'],
            Observation_info['sen_neg']
        ])
        info['Observation'] = Observation_info
        info['message'] = self.msg

        return next_obs, info
    
    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float
    ):
        self.obs = np.array([
            init_pos_press, init_neg_press
        ], dtype=np.float32)

    def set_pid(
        self,
        Kp_pos: float,
        Ki_pos: float,
        Kd_pos: float,
        Kp_neg: float,
        Ki_neg: float,
        Kd_neg: float
    ) -> None:
        self.is_pid = True
        self.pid = PID(
            Kp_pos, Ki_pos, Kd_pos,
            Kp_neg, Ki_neg, Kd_neg,
            freq = self.freq
        )

    def set_anti_windup(
        self,
        Ka  
    ) -> None:
        assert self.is_pid, color("PID controller is not turned on.", "red")
        self.is_anti_windup = True
        self.pid.set_anti_windup(Ka)
        
    def reset_pid(self) -> None:
        self.pid.reset()
    
if __name__ == '__main__':
    env = PneuReal() 

    for _ in range(100):
        obs, info = env.observe(
            ctrl = np.array([-1, 1], dtype=np.float64),
            goal = np.array([120, 130], dtype=np.float64)
        )
        print(obs)
        time.sleep(1)


    


        
