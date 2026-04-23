import sys
#import rospy
from pathlib import Path

from ctypes import CDLL, c_double, POINTER, c_bool

import numpy as np
import math
from typing import Tuple, Optional
from collections import deque

from gymnasium.spaces import Box
from pneu_utils.utils import color
from pneu_env.pid import PID

class PneuSim():
    def __init__(
        self,
        freq: float = 50,
        volume1: float = 0.75,
        volume2: float = 0.4,
        # init_pos_press: float = 120,
        # init_neg_press: float = 80,
        init_pos_press: float = 101.325,
        init_neg_press: float = 101.325,
        delay: float = 0,
        noise: bool = False,
        noise_std: float = 0.2,
        offset_pos: float = 0,
        offset_neg: float = 0,
        scale: bool = False,
        lib_path: Optional[str] = None,
    ):
        if lib_path is None:
            base_dir = Path(__file__).resolve().parent
            candidate_paths = [
                base_dir / "lib" / "pneumatic_simulator.so",
                base_dir / "pneumatic_simulator.so",
            ]
            for candidate in candidate_paths:
                if candidate.is_file():
                    lib_path = str(candidate)
                    break
            else:
                raise FileNotFoundError(
                    "Could not find pneumatic_simulator.so. Tried: "
                    + ", ".join(str(p) for p in candidate_paths)
                )
            print(f"[ INFO] Loaded simulator library from: {lib_path}")
        self.lib = CDLL(lib_path)
        self.lib.set_init_env.argtypes = [c_double, c_double]
        self.lib.set_volume.argtypes = [c_double, c_double]
        self.lib.get_time.restype = c_double
        self.lib.step.argtypes = [POINTER(c_double), c_double]
        self.lib.step.restype = POINTER(c_double)
        self.lib.set_discharge_coeff.argtypes = [c_double for _ in range(4)]
        self.lib.get_mass_flowrate.restype = POINTER(c_double)

        self.lib.solenoid_valve_test.argtypes = [c_double for _ in range(5)]
        self.lib.solenoid_valve_test.restype = c_double
        self.lib.get_mean_mass_flowrate.restype = POINTER(c_double)
        try:
            self.lib.set_logging_c.argtypes = [c_bool]
        except AttributeError:
            pass
        
        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.lib.set_init_env(
            init_pos_press,
            init_neg_press
        )
        self.lib.set_volume(volume1, volume2)

        self.freq = freq

        self.delay = delay
        self.noise = noise
        self.noise_std = noise_std
        self.offset_pos = offset_pos
        self.offset_neg = offset_neg

        obs_buf_len = int(freq*delay + 1)
        self.obs_buf = deque(maxlen=obs_buf_len)
        self.scale = scale
        print(f'[ INFO] Pneumatic Simulator ==> Delay: {delay}')

        # setting PID
        self.is_pid = False
        self.is_anti_windup = False
        self.obs = np.array([101.325, 101.325], dtype=np.float32)
    
    def observe(
        self, 
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325])
    ) -> np.ndarray:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if self.is_pid:
            err = self.pid.get_action(self.obs, goal)
            ctrl += err
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        # Keep action convention consistent with env/predictor.
        ctrl = np.clip(ctrl, -1.0, 1.0)

        if self.is_anti_windup:
            self.pid.anti_windup(
                ctrl=original_ctrl,
                sat_ctrl=ctrl
            )

        if self.scale:
            # [-1, 1] -> [0.7, 1.0]
            ctrl = 0.3 * 0.5 * (ctrl + 1.0) + 0.7
        else:
            # [-1, 1] -> [0, 1]
            ctrl = 0.5 * ctrl + 0.5

        time_step = 1/self.freq
        next_obs = np.array(
            list(self.lib.step((c_double*2)(*list(ctrl)), time_step)[0:3]),
            dtype=np.float64
        )

        self.obs_buf.append(next_obs)
        next_obs = self.obs_buf[0]
        self.obs = next_obs[1:3]

        info = {}
        info['obs_w/o_noise'] = next_obs.copy() 

        if self.noise:
            noise = np.random.normal(0, self.noise_std, 2)
            noise = np.concatenate(([0], noise), axis=0)
            next_obs += noise
            next_obs += np.array([0, self.offset_pos, self.offset_neg])

        Observation_info = dict(
            curr_time = next_obs[0],
            sen_pos = next_obs[1],
            sen_neg = next_obs[2],
            ref_pos = goal[0],
            ref_neg = goal[1],
            ctrl_pos = ctrl[0],
            ctrl_neg = ctrl[1]
        )
        info['Observation'] = Observation_info

        return next_obs, info 
    
    def set_offset(
        self,
        pos_offset: float,
        neg_offset: float
    ):
        self.offset_pos = pos_offset
        self.offset_neg = neg_offset
        
    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float
    ):
        self.lib.time_reset()
        self.lib.set_init_env(
            init_pos_press,
            init_neg_press
        )
        self.obs = np.array([
            init_pos_press, init_neg_press
        ], dtype=np.float32)
    
    def set_volume(self, vol1, vol2):
        self.lib.set_volume(vol1, vol2)
    
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
    
    def set_discharge_coeff(
        self,
        inlet_pump_coeff: float,
        outlet_pump_coeff: float
    ) -> None:
        self.lib.set_discharge_coeff(
            inlet_pump_coeff, outlet_pump_coeff,
            inlet_pump_coeff, outlet_pump_coeff,
        )
    
    def get_mass_flowrate(self):
        return list(self.lib.get_mass_flowrate()[0:5])

    def get_mean_mass_flowrate(self):
        mf = list(self.lib.get_mean_mass_flowrate()[0:4])
        return dict(
            pump_in = mf[0],
            pump_out = mf[1],
            valve_pos = mf[2],
            valve_neg = mf[3]
        )

    def set_logging(self, enable: bool) -> None:
        try:
            self.lib.set_logging_c(enable)
            print(f"[INFO] simulator logging set to: {enable}")
        except AttributeError:
            print(color("[경고] set_logging_c 함수를 찾을 수 없습니다.", "yellow"))

    
    def solenoid_valve(
        self,
        Pin: float,
        Pout: float,
        ctrl: float,
        type: str,
        num: float = 3
    ) -> float:
        return self.lib.solenoid_valve_test(Pin, Pout, ctrl, type, num)

    
if __name__ == '__main__':
    env = PneuSim()
    ctrl = np.array([0.9, 0.9], dtype=np.float64)
    for n in range(10):
        # obs, _ = env.get_obs(ctrl)

        obs, info = env.observe(ctrl)
        print(obs)

        
