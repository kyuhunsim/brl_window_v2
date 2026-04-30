import sys
#.import rospy

from ctypes import CDLL, c_double, POINTER, c_bool, c_int

import numpy as np
import math
from typing import Tuple
from collections import deque

from pneu_utils.utils import get_pkg_path, color
from pneu_env.pid import PID


VALVE_MODEL_PARAM_FIELDS = (
    "flow_multiplier",
    "i_min",
    "i_max",
    "a_max",
    "k_shape",
    "c_k",
    "c_p",
    "c_z",
    "a_bw",
    "beta_bw",
    "gamma_bw",
    "alpha_shape",
    "wn_up",
    "zeta_up",
    "wn_down",
    "zeta_down",
)


class PneuSim():
    """
    lib8 기반 6-밸브 시뮬레이터 래퍼.
    관측: (t, Pch_pos, Pch_neg, P1_pos, P1_neg)
    제어: [u_ch_pos, u_ch_neg, u11_pos, u12_pos, u11_neg, u12_neg] (총 6개)
    """
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
        scale: bool = False,
    ):
        env_pkg_path = get_pkg_path('pneu_env')
        self.lib = CDLL(f'{env_pkg_path}/src/pneu_env/lib8/libpneumatic_simulator.so')

        # C interface
        self.lib.set_init_env_c.argtypes = [c_double, c_double]
        self._has_init_act = False
        try:
            self.lib.set_init_env_act_c.argtypes = [c_double, c_double, c_double, c_double]
            self._has_init_act = True
        except AttributeError:
            self._has_init_act = False
        self.lib.step_c.argtypes = [POINTER(c_double), c_double]
        self.lib.step_c.restype = POINTER(c_double)
        self._has_external_valve_step = False
        try:
            self.lib.step_with_external_valves_c.argtypes = [
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double),
                c_double,
            ]
            self.lib.step_with_external_valves_c.restype = POINTER(c_double)
            self._has_external_valve_step = True
        except AttributeError:
            self._has_external_valve_step = False
        self.lib.get_mass_flowrate_c.restype = POINTER(c_double)
        self._has_mean_mass_flowrate = False
        self._has_valve_param_api = False

        try:
            self.lib.set_volume_c.argtypes = [c_double, c_double]
            self.lib.set_discharge_coeff_c.argtypes = [c_double, c_double, c_double, c_double]
            self.lib.get_time_c.restype = c_double
            self.lib.time_reset_c.argtypes = None
            self.lib.set_logging_c.argtypes = [c_bool]
            self.lib.get_mean_mass_flowrate_c.restype = POINTER(c_double)
            self._has_mean_mass_flowrate = True
        except AttributeError as e:
            print(color(f"[경고] 일부 C 함수가 라이브러리에 없습니다: {e}", "yellow"))

        try:
            self.lib.reset_valve_model_params_c.argtypes = []
            self.lib.set_valve_flow_multiplier_c.argtypes = [c_int, c_double]
            self.lib.set_valve_model_params_c.argtypes = [c_int, POINTER(c_double), c_int]
            self.lib.get_valve_model_params_c.argtypes = [c_int, POINTER(c_double), c_int]
            self._has_valve_param_api = True
        except AttributeError:
            self._has_valve_param_api = False

        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.init_act_pos_press = init_act_pos_press
        self.init_act_neg_press = init_act_neg_press
        if self._has_init_act:
            self.lib.set_init_env_act_c(
                init_pos_press,
                init_neg_press,
                init_act_pos_press,
                init_act_neg_press,
            )
        else:
            self.lib.set_init_env_c(
                init_pos_press,
                init_neg_press
            )
        try:
            self.lib.set_volume_c(volume1, volume2)
        except AttributeError:
            pass

        self.num_solenoids = 6
        self.freq = freq
        self.delay = delay
        self.noise = noise
        self.noise_std = noise_std
        self.offset_pos = offset_pos
        self.offset_neg = offset_neg
        self.offset_act_pos = offset_act_pos
        self.offset_act_neg = offset_act_neg

        obs_buf_len = int(freq * delay + 1)
        self.obs_buf = deque(maxlen=obs_buf_len)
        self.scale = scale
        print(f'[ INFO] Pneumatic Simulator (lib8) ==> Delay: {delay}')

        # setting PID (chamber pos/neg only)
        self.is_pid = False
        self.is_anti_windup = False
        self.obs = np.array([init_pos_press, init_neg_press], dtype=np.float32)

    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325])
    ) -> np.ndarray:
        ctrl = np.asarray(ctrl, dtype=np.float64)

        if ctrl.size == 2:
            ctrl = np.array([ctrl[0], ctrl[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif ctrl.size != self.num_solenoids:
            raise ValueError(
                f"ctrl must be length 2 or {self.num_solenoids}, got {ctrl.size}"
            )

        if self.is_pid:
            goal_posneg = np.asarray(goal, dtype=np.float64)
            if goal_posneg.size >= 2:
                goal_posneg = goal_posneg[:2]
            else:
                goal_posneg = np.array([101.325, 101.325], dtype=np.float64)
            err = self.pid.get_action(self.obs, goal_posneg)
            ctrl = ctrl.copy()
            ctrl[:2] += err
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        # Keep action convention consistent across lib2/lib5/lib8 and predictor.
        # ctrl = np.clip(ctrl, -1.0, 1.0)

        if self.is_anti_windup:
            self.pid.anti_windup(
                ctrl=original_ctrl[:2],
                sat_ctrl=ctrl[:2]
            )

        if self.scale:
            # # [-1, 1] -> [0.7, 1.0]
            # ctrl = 0.3 * 0.5 * (ctrl + 1.0) + 0.7
            # [-1, 1] -> [0.8, 1.0]
            # ctrl = 0.2 * 0.5 * (ctrl + 1.0) + 0.8
            ctrl=ctrl
        else:
            # [-1, 1] -> [0, 1]
            # ctrl = 0.5 * ctrl + 0.5
            ctrl=ctrl

        time_step = 1 / self.freq
        next_obs = np.array(
            list(self.lib.step_c((c_double * self.num_solenoids)(*list(ctrl)), time_step)[0:5]),
            dtype=np.float64
        )

        self.obs_buf.append(next_obs)
        next_obs = self.obs_buf[0]
        self.obs = next_obs[1:3]

        info = {}
        info['obs_w/o_noise'] = next_obs.copy()

        if self.noise:
            noise = np.random.normal(0, self.noise_std, 4)
            noise = np.concatenate(([0], noise), axis=0)
            next_obs += noise
            next_obs += np.array([
                0,
                self.offset_pos,
                self.offset_neg,
                self.offset_act_pos,
                self.offset_act_neg
            ])

        ref_act_pos = float("nan")
        ref_act_neg = float("nan")
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float64)
            if goal.size >= 4:
                ref_act_pos = float(goal[2])
                ref_act_neg = float(goal[3])

        Observation_info = dict(
            curr_time=next_obs[0],
            sen_pos=next_obs[1],
            sen_neg=next_obs[2],
            P1_pos=next_obs[3],
            P1_neg=next_obs[4],
            ref_pos=goal[0] if goal is not None and len(goal) >= 2 else 101.325,
            ref_neg=goal[1] if goal is not None and len(goal) >= 2 else 101.325,
            ref_act_pos=ref_act_pos,
            ref_act_neg=ref_act_neg,
            ctrl_pos=ctrl[0],
            ctrl_neg=ctrl[1],
            act_pos_ctrl1=ctrl[2],
            act_pos_ctrl2=ctrl[3],
            act_neg_ctrl1=ctrl[4],
            act_neg_ctrl2=ctrl[5],
        )
        info['Observation'] = Observation_info

        return next_obs, info

    def observe_with_external_valves(
        self,
        ctrl_unit: np.ndarray,
        valve_pin: np.ndarray,
        valve_pout: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325]),
    ) -> Tuple[np.ndarray, dict]:
        """
        Hybrid valve replay step.

        - ctrl_unit: 6 valve commands already in [0, 1]
        - valve_pin / valve_pout: real pressure pairs [kPa] for each valve
        - pressure states are still advanced by lib8 RK integration
        """
        if not self._has_external_valve_step:
            raise RuntimeError("lib8 does not expose step_with_external_valves_c")

        ctrl_unit = np.asarray(ctrl_unit, dtype=np.float64).reshape(-1)
        if ctrl_unit.size == 2:
            ctrl_unit = np.array([ctrl_unit[0], ctrl_unit[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif ctrl_unit.size != self.num_solenoids:
            raise ValueError(
                f"ctrl_unit must be length 2 or {self.num_solenoids}, got {ctrl_unit.size}"
            )
        ctrl_unit = np.clip(ctrl_unit, 0.0, 1.0)

        valve_pin = np.asarray(valve_pin, dtype=np.float64).reshape(-1)
        valve_pout = np.asarray(valve_pout, dtype=np.float64).reshape(-1)
        if valve_pin.size != self.num_solenoids or valve_pout.size != self.num_solenoids:
            raise ValueError(
                "valve_pin and valve_pout must each have "
                f"{self.num_solenoids} values, got {valve_pin.size} and {valve_pout.size}"
            )

        time_step = 1 / self.freq
        next_obs = np.array(
            list(
                self.lib.step_with_external_valves_c(
                    (c_double * self.num_solenoids)(*list(ctrl_unit)),
                    (c_double * self.num_solenoids)(*list(valve_pin)),
                    (c_double * self.num_solenoids)(*list(valve_pout)),
                    time_step,
                )[0:5]
            ),
            dtype=np.float64,
        )

        self.obs_buf.append(next_obs)
        next_obs = self.obs_buf[0]
        self.obs = next_obs[1:3]

        info = {}
        info['obs_w/o_noise'] = next_obs.copy()

        if self.noise:
            noise = np.random.normal(0, self.noise_std, 4)
            noise = np.concatenate(([0], noise), axis=0)
            next_obs += noise
            next_obs += np.array([
                0,
                self.offset_pos,
                self.offset_neg,
                self.offset_act_pos,
                self.offset_act_neg
            ])

        ref_act_pos = float("nan")
        ref_act_neg = float("nan")
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float64)
            if goal.size >= 4:
                ref_act_pos = float(goal[2])
                ref_act_neg = float(goal[3])

        Observation_info = dict(
            curr_time=next_obs[0],
            sen_pos=next_obs[1],
            sen_neg=next_obs[2],
            P1_pos=next_obs[3],
            P1_neg=next_obs[4],
            ref_pos=goal[0] if goal is not None and len(goal) >= 2 else 101.325,
            ref_neg=goal[1] if goal is not None and len(goal) >= 2 else 101.325,
            ref_act_pos=ref_act_pos,
            ref_act_neg=ref_act_neg,
            ctrl_pos=ctrl_unit[0],
            ctrl_neg=ctrl_unit[1],
            act_pos_ctrl1=ctrl_unit[2],
            act_pos_ctrl2=ctrl_unit[3],
            act_neg_ctrl1=ctrl_unit[4],
            act_neg_ctrl2=ctrl_unit[5],
        )
        info['Observation'] = Observation_info

        return next_obs, info

    def set_offset(
        self,
        pos_offset: float,
        neg_offset: float,
        act_pos_offset: float = 0.0,
        act_neg_offset: float = 0.0,
    ):
        self.offset_pos = pos_offset
        self.offset_neg = neg_offset
        self.offset_act_pos = act_pos_offset
        self.offset_act_neg = act_neg_offset

    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
    ):
        try:
            self.lib.time_reset_c()
        except AttributeError:
            pass
        if self._has_init_act:
            self.lib.set_init_env_act_c(
                init_pos_press,
                init_neg_press,
                init_act_pos_press,
                init_act_neg_press,
            )
        else:
            self.lib.set_init_env_c(
                init_pos_press,
                init_neg_press
            )
        self.obs = np.array([
            init_pos_press, init_neg_press
        ], dtype=np.float32)

    def set_volume(self, vol1, vol2):
        try:
            self.lib.set_volume_c(vol1, vol2)
        except AttributeError:
            print(color("[경고] set_volume_c 함수를 찾을 수 없습니다.", "yellow"))

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
            freq=self.freq
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
        try:
            self.lib.set_discharge_coeff_c(
                inlet_pump_coeff, outlet_pump_coeff,
                inlet_pump_coeff, outlet_pump_coeff,
            )
        except AttributeError:
            print(color("[경고] set_discharge_coeff_c 함수를 찾을 수 없습니다.", "yellow"))

    def reset_valve_model_params(self) -> None:
        if not self._has_valve_param_api:
            print(color("[경고] valve param C API를 찾을 수 없습니다.", "yellow"))
            return
        self.lib.reset_valve_model_params_c()

    def get_valve_model_params(self, valve_idx: int) -> dict:
        if not self._has_valve_param_api:
            print(color("[경고] valve param C API를 찾을 수 없습니다.", "yellow"))
            return {}
        if valve_idx < 0 or valve_idx >= self.num_solenoids:
            raise ValueError(f"valve_idx must be in [0, {self.num_solenoids - 1}], got {valve_idx}")

        values = (c_double * len(VALVE_MODEL_PARAM_FIELDS))()
        self.lib.get_valve_model_params_c(
            int(valve_idx),
            values,
            len(VALVE_MODEL_PARAM_FIELDS),
        )
        return {
            name: float(values[i])
            for i, name in enumerate(VALVE_MODEL_PARAM_FIELDS)
        }

    def set_valve_flow_multiplier(self, valve_idx: int, flow_multiplier: float) -> None:
        if not self._has_valve_param_api:
            print(color("[경고] valve param C API를 찾을 수 없습니다.", "yellow"))
            return
        if valve_idx < 0 or valve_idx >= self.num_solenoids:
            raise ValueError(f"valve_idx must be in [0, {self.num_solenoids - 1}], got {valve_idx}")

        self.lib.set_valve_flow_multiplier_c(int(valve_idx), float(flow_multiplier))

    def set_valve_model_params(self, valve_idx: int, **kwargs) -> None:
        if not self._has_valve_param_api:
            print(color("[경고] valve param C API를 찾을 수 없습니다.", "yellow"))
            return
        if valve_idx < 0 or valve_idx >= self.num_solenoids:
            raise ValueError(f"valve_idx must be in [0, {self.num_solenoids - 1}], got {valve_idx}")

        invalid_keys = sorted(set(kwargs) - set(VALVE_MODEL_PARAM_FIELDS))
        if invalid_keys:
            raise ValueError(
                f"Unsupported valve param keys: {invalid_keys}. "
                f"Expected subset of {VALVE_MODEL_PARAM_FIELDS}"
            )

        params = self.get_valve_model_params(valve_idx)
        params.update({key: float(value) for key, value in kwargs.items()})
        serialized = [params[name] for name in VALVE_MODEL_PARAM_FIELDS]
        self.lib.set_valve_model_params_c(
            int(valve_idx),
            (c_double * len(serialized))(*serialized),
            len(serialized),
        )

    def get_mass_flowrate(self):
        """
        Returns 10 instantaneous mass flowrates [kg/s] in lib6-compatible order:
          [0] m_po1, [1] m_pi1, [2] m_po2, [3] m_pi2,
          [4] m_sv_cp, [5] m_sv_cn,
          [6] m_sv11p, [7] m_sv12p, [8] m_sv11n, [9] m_sv12n
        """
        return list(self.lib.get_mass_flowrate_c()[0:10])

    def get_mean_mass_flowrate(self):
        """
        Returns mean mass flowrates [kg/s] over the latest observe()/step interval.
        Keys follow the legacy tuner contract.
        """
        if self._has_mean_mass_flowrate:
            mf = list(self.lib.get_mean_mass_flowrate_c()[0:4])
            return dict(
                pump_in=float(mf[0]),
                pump_out=float(mf[1]),
                valve_pos=float(mf[2]),
                valve_neg=float(mf[3]),
            )

        mf = self.get_mass_flowrate_dict()
        return dict(
            pump_in=float(mf["pump_in"]),
            pump_out=float(mf["pump_out"]),
            valve_pos=float(mf["chamber_pos_valve"]),
            valve_neg=float(mf["chamber_neg_valve"]),
        )

    def get_mass_flowrate_dict(self):
        """Named accessors with both raw valve flows and legacy aggregate aliases."""
        mf = self.get_mass_flowrate()
        m_po1 = float(mf[0])
        m_pi1 = float(mf[1])
        m_po2 = float(mf[2])
        m_pi2 = float(mf[3])
        m_sv_cp = float(mf[4])
        m_sv_cn = float(mf[5])
        m_sv11p = float(mf[6])
        m_sv12p = float(mf[7])
        m_sv11n = float(mf[8])
        m_sv12n = float(mf[9])

        return dict(
            m_po1=m_po1,
            m_pi1=m_pi1,
            m_po2=m_po2,
            m_pi2=m_pi2,
            m_sv_cp=m_sv_cp,
            m_sv_cn=m_sv_cn,
            m_sv11p=m_sv11p,
            m_sv12p=m_sv12p,
            m_sv11n=m_sv11n,
            m_sv12n=m_sv12n,

            # Legacy aliases used by older scripts.
            pump_out=m_po1 + m_po2,
            pump_in=m_pi1 + m_pi2,
            chamber_pos_valve=m_sv_cp,
            chamber_neg_valve=m_sv_cn,
            act_pos_net_in=m_sv11p - m_sv12p,
            act_neg_net_in=m_sv11n - m_sv12n,
            act_pos_in=m_sv11p,
            act_pos_out=m_sv12p,
            act_neg_in=m_sv11n,
            act_neg_out=m_sv12n,

            # 6-flow alias mapping requested by hardware integration:
            # flow1: chamber pos valve, flow2: act pos in, flow3: act pos out,
            # flow4: chamber neg valve, flow5: act neg out, flow6: act neg in.
            flow1=m_sv_cp,
            flow2=m_sv11p,
            flow3=m_sv12p,
            flow4=m_sv_cn,
            flow5=m_sv12n,
            flow6=m_sv11n,
        )

    def set_logging(self, enable: bool) -> None:
        try:
            self.lib.set_logging_c(enable)
            print(f"[INFO] lib8 detailed logging set to: {enable}")
        except AttributeError:
            print(color("[경고] set_logging_c 함수를 찾을 수 없습니다.", "yellow"))


if __name__ == '__main__':
    env = PneuSim()
    env.set_logging(True)
    ctrl = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for n in range(10):
        obs, info = env.observe(ctrl)
        print(obs)
