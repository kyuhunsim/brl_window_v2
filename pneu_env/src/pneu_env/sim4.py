from __future__ import annotations

from collections import deque
from ctypes import CDLL, POINTER, c_bool, c_double
from pathlib import Path

import numpy as np

from pneu_env.pid import ActuatorPressurePID
from pneu_utils.utils import get_pkg_path


class PneuSim:
    """lib4 simulator: lib3 valve/pump model plus soft-actuator motion."""

    def __init__(
        self,
        freq: float = 50,
        volume1: float = 0.75,
        volume2: float = 0.4,
        init_pos_press: float = 101.325,
        init_neg_press: float = 101.325,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
        init_length: float | None = None,
        init_velocity: float = 0.0,
        actuator_initial_length: float = 0.02,
        actuator_dimension_D: float = 0.02,
        actuator_num_folds: float = 2.0,
        actuator_shaft_radius: float = 0.005,
        actuator_rod_mass: float = 5.0,
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
        lib_path = env_pkg_path / "src/pneu_env/lib4/libpneumatic_simulator.so"
        if not lib_path.is_file():
            raise FileNotFoundError(f"Could not find lib4 simulator library: {lib_path}")

        self.lib = CDLL(str(lib_path))
        print(f"[ INFO] Loaded pneumatic simulator library from: {lib_path}")

        self.lib.initialize_parameters.argtypes = [c_double for _ in range(5)]
        self.lib.get_initial_length.restype = c_double
        self.lib.get_contraction_ratio.argtypes = [c_double]
        self.lib.get_contraction_ratio.restype = c_double
        self.lib.get_volume_pos.argtypes = [c_double]
        self.lib.get_volume_pos.restype = c_double
        self.lib.get_volume_neg.argtypes = [c_double]
        self.lib.get_volume_neg.restype = c_double
        self.lib.set_init_env_c.argtypes = [c_double for _ in range(4)]
        self.lib.set_init_state_c.argtypes = [c_double for _ in range(6)]
        self.lib.set_volume_c.argtypes = [c_double, c_double]
        self.lib.get_time_c.restype = c_double
        self.lib.step_c.argtypes = [POINTER(c_double), c_double]
        self.lib.step_c.restype = POINTER(c_double)
        self.lib.set_discharge_coeff_c.argtypes = [c_double for _ in range(4)]
        self.lib.get_mass_flowrate_c.restype = POINTER(c_double)
        self.lib.get_mean_mass_flowrate_c.restype = POINTER(c_double)
        self.lib.get_valve_debug_c.restype = POINTER(c_double)
        self.lib.time_reset_c.argtypes = []
        self.lib.set_logging_c.argtypes = [c_bool]

        self.lib.initialize_parameters(
            actuator_initial_length,
            actuator_dimension_D,
            actuator_num_folds,
            actuator_shaft_radius,
            actuator_rod_mass,
        )
        self.initial_length = self.lib.get_initial_length()
        if init_length is None:
            init_length = self.initial_length

        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.init_act_pos_press = init_act_pos_press
        self.init_act_neg_press = init_act_neg_press
        self.init_length = init_length
        self.init_velocity = init_velocity

        self.lib.set_init_state_c(
            init_pos_press,
            init_neg_press,
            init_act_pos_press,
            init_act_neg_press,
            init_length,
            init_velocity,
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
            [
                init_pos_press,
                init_neg_press,
                init_act_pos_press,
                init_act_neg_press,
                init_length,
                init_velocity,
            ],
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
        ctrl = np.asarray(ctrl, dtype=np.float64).copy()
        goal = np.asarray(goal, dtype=np.float64)
        if ctrl.shape != (6,):
            raise ValueError(f"sim4 expects 6 control inputs, got shape {ctrl.shape}")
        if goal.shape != (2,):
            raise ValueError(f"sim4 expects 2 actuator reference inputs, got shape {goal.shape}")

        if self.is_pid:
            pid_delta = self.pid.get_action(obs=self.obs[2:4], ref=goal)
            ctrl[2:6] += pid_delta
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        ctrl = np.clip(ctrl, 0.0, 1.0)

        if self.is_anti_windup:
            self.pid.anti_windup(ctrl=original_ctrl[2:6], sat_ctrl=ctrl[2:6])

        if self.scale:
            ctrl = 0.85 + 0.15 * ctrl

        time_step = 1 / self.freq
        next_obs = np.array(
            list(self.lib.step_c((c_double * 6)(*list(ctrl)), time_step)[0:7]),
            dtype=np.float64,
        )

        self.obs_buf.append(next_obs)
        next_obs = self.obs_buf[0].copy()
        self.obs = next_obs[1:7].copy()
        info = {"obs_w/o_noise": next_obs.copy()}

        if self.noise:
            next_obs[1:5] += np.random.normal(0, self.noise_std, 4)
            next_obs[1:5] += np.array(
                [self.offset_pos, self.offset_neg, self.offset_act_pos, self.offset_act_neg]
            )

        info["Observation"] = dict(
            curr_time=next_obs[0],
            sen_pos=next_obs[1],
            sen_neg=next_obs[2],
            sen_act_pos=next_obs[3],
            sen_act_neg=next_obs[4],
            sen_length=next_obs[5],
            sen_velocity=next_obs[6],
            contraction_ratio=self.lib.get_contraction_ratio(next_obs[5]),
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
        init_length: float | None = None,
        init_velocity: float = 0.0,
    ) -> None:
        if init_length is None:
            init_length = self.initial_length
        self.lib.set_init_state_c(
            init_pos_press,
            init_neg_press,
            init_act_pos_press,
            init_act_neg_press,
            init_length,
            init_velocity,
        )
        self.obs = np.array(
            [
                init_pos_press,
                init_neg_press,
                init_act_pos_press,
                init_act_neg_press,
                init_length,
                init_velocity,
            ],
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

    def get_mean_mass_flowrate(self) -> list[float]:
        return list(self.lib.get_mean_mass_flowrate_c()[0:4])

    def get_valve_debug(self) -> dict[str, float]:
        names = [
            "u_eff",
            "current",
            "state_curr",
            "z",
            "force_net",
            "area_eff",
            "q_static_lpm",
            "q_pred_lpm",
            "mdot",
        ]
        valve_names = [
            "ch_pos",
            "ch_neg",
            "act_pos_in",
            "act_pos_out",
            "act_neg_in",
            "act_neg_out",
        ]
        values = list(self.lib.get_valve_debug_c()[0:54])
        debug = {}
        for valve_idx, valve_name in enumerate(valve_names):
            offset = valve_idx * len(names)
            for name_idx, name in enumerate(names):
                debug[f"{valve_name}_{name}"] = values[offset + name_idx]
        return debug
