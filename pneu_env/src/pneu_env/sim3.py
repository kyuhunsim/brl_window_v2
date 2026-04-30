from ctypes import CDLL, POINTER, c_bool, c_double
from pathlib import Path
from collections import deque

import numpy as np

from pneu_utils.utils import get_pkg_path
from pneu_env.pid import ActuatorPressurePID


class PneuSim:
    def __init__(
        self,
        freq: float = 50,
        volume1: float = 0.75,
        volume2: float = 0.4,
        init_pos_press: float = 101.325,
        init_neg_press: float = 101.325,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
        delay: float = 0,
        noise: bool = False,
        noise_std: float = 0.2,
        offset_pos: float = 0,
        offset_neg: float = 0,
        offset_act_pos: float = 0,
        offset_act_neg: float = 0,
        scale: bool = True,
    ):
        env_pkg_path = Path(get_pkg_path("pneu_env"))
        lib_path = env_pkg_path / "src/pneu_env/lib3/libpneumatic_simulator.so"
        if not lib_path.is_file():
            raise FileNotFoundError(f"Could not find lib3 simulator library: {lib_path}")

        self.lib = CDLL(str(lib_path))
        print(f"[ INFO] Loaded pneumatic simulator library from: {lib_path}")

        self.lib.set_init_env_c.argtypes = [c_double, c_double]
        self.lib.set_init_env_act_c.argtypes = [c_double, c_double, c_double, c_double]
        self.lib.set_volume_c.argtypes = [c_double, c_double]
        self.lib.get_time_c.restype = c_double
        self.lib.step_c.argtypes = [POINTER(c_double), c_double]
        self.lib.step_c.restype = POINTER(c_double)
        self.lib.set_discharge_coeff_c.argtypes = [c_double for _ in range(4)]
        self.lib.get_mass_flowrate_c.restype = POINTER(c_double)
        self.lib.time_reset_c.argtypes = []
        self.lib.set_logging_c.argtypes = [c_bool]

        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.init_act_pos_press = init_act_pos_press
        self.init_act_neg_press = init_act_neg_press

        self.lib.set_init_env_act_c(
            init_pos_press,
            init_neg_press,
            init_act_pos_press,
            init_act_neg_press,
        )
        self.lib.set_volume_c(volume1, volume2)

        self.freq = freq
        self.delay = delay
        self.noise = noise
        self.noise_std = noise_std
        self.offset_pos = offset_pos
        self.offset_neg = offset_neg
        self.offset_act_pos = offset_act_pos
        self.offset_act_neg = offset_act_neg
        self.scale = scale

        obs_buf_len = int(freq * delay + 1)
        self.obs_buf = deque(maxlen=obs_buf_len)
        self.obs = np.array(
            [init_pos_press, init_neg_press, init_act_pos_press, init_act_neg_press],
            dtype=np.float64,
        )
        self.is_pid = False
        self.is_anti_windup = False

        print(f"[ INFO] Pneumatic Simulator ==> Delay: {delay}")

    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325]),
    ) -> tuple[np.ndarray, dict]:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if ctrl.shape != (6,):
            raise ValueError(f"sim3 expects 6 control inputs, got shape {ctrl.shape}")
        if goal.shape != (2,):
            raise ValueError(f"sim3 expects 2 actuator reference inputs, got shape {goal.shape}")

        if self.is_pid:
            pid_delta = self.pid.get_action(
                obs=self.obs[2:4],
                ref=goal,
            )
            ctrl[2:6] += pid_delta
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        ctrl = np.clip(ctrl, -1, 1)

        if self.is_anti_windup:
            self.pid.anti_windup(
                ctrl=original_ctrl[2:6],
                sat_ctrl=ctrl[2:6],
            )

        if self.scale:
            ctrl = 0.3 * 0.5 * (ctrl + 1) + 0.7
        else:
            ctrl = 0.5 * ctrl + 0.5

        time_step = 1 / self.freq
        next_obs = np.array(
            list(self.lib.step_c((c_double * 6)(*list(ctrl)), time_step)[0:5]),
            dtype=np.float64,
        )

        self.obs_buf.append(next_obs)
        next_obs = self.obs_buf[0]
        self.obs = next_obs[1:5]

        info = {"obs_w/o_noise": next_obs.copy()}

        if self.noise:
            noise = np.random.normal(0, self.noise_std, 4)
            noise = np.concatenate(([0], noise), axis=0)
            next_obs += noise
            next_obs += np.array([
                0,
                self.offset_pos,
                self.offset_neg,
                self.offset_act_pos,
                self.offset_act_neg,
            ])

        info["Observation"] = dict(
            curr_time=next_obs[0],
            sen_pos=next_obs[1],
            sen_neg=next_obs[2],
            sen_act_pos=next_obs[3],
            sen_act_neg=next_obs[4],
            ref_pos=101.325,
            ref_neg=101.325,
            ref_act_pos=goal[0],
            ref_act_neg=goal[1],
            ctrl_pos=ctrl[0],
            ctrl_neg=ctrl[1],
            ctrl_act_pos_in=ctrl[2],
            ctrl_act_pos_out=ctrl[3],
            ctrl_act_neg_in=ctrl[4],
            ctrl_act_neg_out=ctrl[5],
        )

        return next_obs, info

    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
    ) -> None:
        self.lib.time_reset_c()
        self.lib.set_init_env_act_c(
            init_pos_press,
            init_neg_press,
            init_act_pos_press,
            init_act_neg_press,
        )
        self.obs = np.array(
            [init_pos_press, init_neg_press, init_act_pos_press, init_act_neg_press],
            dtype=np.float64,
        )
        self.obs_buf.clear()

    def set_volume(self, vol1: float, vol2: float) -> None:
        self.lib.set_volume_c(vol1, vol2)

    def set_pid(
        self,
        Kp_act_pos_in: float,
        Ki_act_pos_in: float,
        Kd_act_pos_in: float,
        Kp_act_pos_out: float,
        Ki_act_pos_out: float,
        Kd_act_pos_out: float,
        Kp_act_neg_in: float,
        Ki_act_neg_in: float,
        Kd_act_neg_in: float,
        Kp_act_neg_out: float,
        Ki_act_neg_out: float,
        Kd_act_neg_out: float,
    ) -> None:
        self.is_pid = True
        self.pid = ActuatorPressurePID(
            Kp_act_pos_in,
            Ki_act_pos_in,
            Kd_act_pos_in,
            Kp_act_pos_out,
            Ki_act_pos_out,
            Kd_act_pos_out,
            Kp_act_neg_in,
            Ki_act_neg_in,
            Kd_act_neg_in,
            Kp_act_neg_out,
            Ki_act_neg_out,
            Kd_act_neg_out,
            freq=self.freq,
        )

    def set_anti_windup(self, Ka: float) -> None:
        if not self.is_pid:
            raise RuntimeError("PID controller is not turned on.")
        self.is_anti_windup = True
        self.pid.set_anti_windup(Ka)

    def reset_pid(self) -> None:
        if self.is_pid:
            self.pid.reset()

    def set_discharge_coeff(
        self,
        inlet_pump_coeff: float,
        outlet_pump_coeff: float,
    ) -> None:
        self.lib.set_discharge_coeff_c(
            inlet_pump_coeff,
            outlet_pump_coeff,
            inlet_pump_coeff,
            outlet_pump_coeff,
        )

    def set_logging(self, enable: bool) -> None:
        self.lib.set_logging_c(enable)

    def get_mass_flowrate(self) -> list[float]:
        return list(self.lib.get_mass_flowrate_c()[0:10])


if __name__ == "__main__":
    env = PneuSim()
    ctrl = np.array([0.9, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(10):
        obs, _ = env.observe(ctrl)
        print(obs)
