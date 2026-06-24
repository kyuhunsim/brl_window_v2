from __future__ import annotations

from collections import deque
from ctypes import CDLL, POINTER, c_bool, c_double
from pathlib import Path
from typing import Literal

import numpy as np

from pneu_env.pid import ActuatorPressurePID
from pneu_utils.utils import get_pkg_path


DEFAULT_ENCODER_PITCH_RADIUS_M = 22.92e-3
DEFAULT_ENCODER_GEAR_RATIO = 1.0
DEFAULT_ENCODER_DISPLACEMENT_SIGN = -1.0
DEFAULT_ACTUATOR_SERIES_COUNT = 2.0
DEFAULT_ACTUATOR_MAX_CONTRACTION_M = 32e-3


class PneuSim:
    """lib4 simulator: valve/pump pneumatic model plus soft-actuator motion."""

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
        actuator_series_count: float | None = None,
        actuator_max_contraction: float = DEFAULT_ACTUATOR_MAX_CONTRACTION_M,
        actuator_min_total_length: float | None = None,
        actuator_hard_min_cell_length: float | None = None,
        actuator_shaft_radius: float = 0.005,
        actuator_rod_mass: float = 10.3,
        encoder_pitch_radius: float = DEFAULT_ENCODER_PITCH_RADIUS_M,
        encoder_gear_ratio: float = DEFAULT_ENCODER_GEAR_RATIO,
        encoder_displacement_sign: float = DEFAULT_ENCODER_DISPLACEMENT_SIGN,
        encoder_angle_unit: Literal["deg", "rad"] = "deg",
        encoder_angular_velocity_scale: float = 1.0,
        lib_path: str | Path | None = None,
        delay: float = 0,
        noise: bool = False,
        noise_std: float = 0.2,
        offset_pos: float = 0,
        offset_neg: float = 0,
        offset_act_pos: float = 0,
        offset_act_neg: float = 0,
        scale: bool = False,
        scale_low: float = 0.85,
        scale_high: float = 1.0,
    ):
        if lib_path is None:
            pkg_path = Path(get_pkg_path("pneu_env"))
            lib_path = pkg_path / "src/pneu_env/lib4/libpneumatic_simulator.so"
        else:
            lib_path = Path(lib_path)
        if not lib_path.is_file():
            raise FileNotFoundError(f"Could not find lib4 simulator library: {lib_path}")

        self.lib = CDLL(str(lib_path))
        print(f"[ INFO] Loaded pneumatic simulator library from: {lib_path}")

        self.lib.initialize_parameters.argtypes = [c_double for _ in range(5)]
        self.lib.get_initial_length.restype = c_double
        self.lib.set_actuator_min_length_c.argtypes = [c_double]
        self.lib.get_min_length_c.restype = c_double
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
        self.lib.set_leak_coefficients_c.argtypes = [c_double for _ in range(3)]
        self.lib.get_mass_flowrate_c.restype = POINTER(c_double)
        self.lib.get_mean_mass_flowrate_c.restype = POINTER(c_double)
        self.lib.get_valve_debug_c.restype = POINTER(c_double)
        self.lib.time_reset_c.argtypes = []
        self.lib.set_logging_c.argtypes = [c_bool]

        if actuator_series_count is None:
            actuator_series_count = actuator_num_folds
        self.actuator_series_count = float(actuator_series_count)
        self.actuator_cell_initial_length = float(actuator_initial_length)
        self.actuator_total_initial_length = (
            self.actuator_cell_initial_length * self.actuator_series_count
        )
        if actuator_min_total_length is None:
            actuator_min_total_length = (
                self.actuator_total_initial_length - float(actuator_max_contraction)
            )
        else:
            actuator_max_contraction = (
                self.actuator_total_initial_length - float(actuator_min_total_length)
            )
        self.actuator_min_total_length = float(actuator_min_total_length)
        self.actuator_max_contraction = float(actuator_max_contraction)
        self.actuator_stroke_min_cell_length = (
            self.actuator_min_total_length / self.actuator_series_count
        )
        if actuator_hard_min_cell_length is None:
            actuator_hard_min_cell_length = 0.0
        self.actuator_hard_min_cell_length = float(actuator_hard_min_cell_length)
        self.encoder_pitch_radius = float(encoder_pitch_radius)
        self.encoder_gear_ratio = float(encoder_gear_ratio)
        self.encoder_displacement_sign = float(encoder_displacement_sign)
        self.encoder_angle_unit = str(encoder_angle_unit)
        self.encoder_angular_velocity_scale = float(encoder_angular_velocity_scale)
        if self.encoder_angle_unit not in ("deg", "rad"):
            raise ValueError(
                f"encoder_angle_unit must be 'deg' or 'rad', got {self.encoder_angle_unit!r}"
            )
        if self.encoder_pitch_radius <= 0.0:
            raise ValueError("encoder_pitch_radius must be positive")
        if self.encoder_gear_ratio <= 0.0:
            raise ValueError("encoder_gear_ratio must be positive")
        if self.actuator_series_count <= 0.0:
            raise ValueError("actuator_series_count must be positive")
        if self.actuator_cell_initial_length <= 0.0:
            raise ValueError("actuator_initial_length must be positive")
        if self.actuator_total_initial_length <= 0.0:
            raise ValueError("actuator total initial length must be positive")
        if self.actuator_max_contraction <= 0.0:
            raise ValueError("actuator_max_contraction must be positive")
        if self.actuator_min_total_length < 0.0:
            raise ValueError("actuator_min_total_length must be non-negative")
        if self.actuator_stroke_min_cell_length >= self.actuator_cell_initial_length:
            raise ValueError(
                "actuator_min_total_length leaves no available stroke for the cell coordinate"
            )
        if self.actuator_hard_min_cell_length < 0.0:
            raise ValueError("actuator_hard_min_cell_length must be non-negative")
        if self.actuator_hard_min_cell_length >= self.actuator_cell_initial_length:
            raise ValueError(
                "actuator_hard_min_cell_length must be smaller than actuator_initial_length"
            )

        self.lib.initialize_parameters(
            actuator_initial_length,
            actuator_dimension_D,
            actuator_num_folds,
            actuator_shaft_radius,
            actuator_rod_mass,
        )
        self.lib.set_actuator_min_length_c(self.actuator_hard_min_cell_length)
        self.initial_length = self.lib.get_initial_length()
        self.minimum_length = self.lib.get_min_length_c()
        if init_length is None:
            init_length = self.initial_length
        init_length = self.clamp_cell_length(init_length)

        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.init_act_pos_press = init_act_pos_press
        self.init_act_neg_press = init_act_neg_press
        self.init_length = init_length
        self.init_total_length = self.cell_length_to_total_length(init_length)
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
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        if self.scale_high <= self.scale_low:
            raise ValueError(
                f"scale_high must be larger than scale_low, got "
                f"{self.scale_low}..{self.scale_high}"
            )

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

    def _angle_delta_to_rad(self, angle_delta: np.ndarray | float) -> np.ndarray | float:
        if self.encoder_angle_unit == "deg":
            return np.deg2rad(angle_delta)
        return angle_delta

    def angle_to_displacement(
        self,
        angle: np.ndarray | float,
        angle_initial: float,
    ) -> np.ndarray | float:
        """Convert encoder angle to total relative actuator/rod displacement [m].

        The default sign follows the current calibration: decreasing encoder
        angle is positive contraction / rod-lift displacement.
        """
        angle_delta = np.asarray(angle, dtype=np.float64) - float(angle_initial)
        displacement = (
            self.encoder_displacement_sign
            * self.encoder_pitch_radius
            * self.encoder_gear_ratio
            * self._angle_delta_to_rad(angle_delta)
        )
        if np.isscalar(angle):
            return float(displacement)
        return displacement

    def angle_to_length_delta(
        self,
        angle: np.ndarray | float,
        angle_initial: float,
    ) -> np.ndarray | float:
        """Convert encoder angle to total actuator length change [m].

        Positive means the actuator total length increases. With the calibrated
        encoder sign, this is the negative of contraction displacement.
        """
        length_delta = -np.asarray(self.angle_to_displacement(angle, angle_initial), dtype=np.float64)
        if np.isscalar(angle):
            return float(length_delta)
        return length_delta

    def total_to_cell_displacement(self, total_displacement: np.ndarray | float) -> np.ndarray | float:
        """Map total actuator displacement to the per-cell coordinate used by lib4."""
        value = np.asarray(total_displacement, dtype=np.float64) / self.actuator_series_count
        if np.isscalar(total_displacement):
            return float(value)
        return value

    def cell_to_total_displacement(self, cell_displacement: np.ndarray | float) -> np.ndarray | float:
        """Map lib4 per-cell displacement to total actuator displacement."""
        value = np.asarray(cell_displacement, dtype=np.float64) * self.actuator_series_count
        if np.isscalar(cell_displacement):
            return float(value)
        return value

    def cell_length_to_total_length(self, cell_length: np.ndarray | float) -> np.ndarray | float:
        """Convert lib4 cell length to total actuator active length [m]."""
        return self.cell_to_total_displacement(cell_length)

    def total_length_to_cell_length(self, total_length: np.ndarray | float) -> np.ndarray | float:
        """Convert total actuator active length to lib4 cell length [m]."""
        return self.total_to_cell_displacement(total_length)

    def clamp_total_length(self, total_length: np.ndarray | float) -> np.ndarray | float:
        """Clamp total actuator length to the calibrated 40/32/8 mm stroke range."""
        value = np.clip(
            np.asarray(total_length, dtype=np.float64),
            self.actuator_min_total_length,
            self.actuator_total_initial_length,
        )
        if np.isscalar(total_length):
            return float(value)
        return value

    def clamp_cell_length(self, cell_length: np.ndarray | float) -> np.ndarray | float:
        """Clamp lib4 cell length to the optional hard-stop range.

        The 32 mm external stroke limit is handled in total coordinates. By
        default lib4 keeps its original 0..L0 cell-coordinate integration range.
        """
        value = np.clip(
            np.asarray(cell_length, dtype=np.float64),
            self.minimum_length if hasattr(self, "minimum_length") else self.actuator_hard_min_cell_length,
            self.initial_length if hasattr(self, "initial_length") else self.actuator_cell_initial_length,
        )
        if np.isscalar(cell_length):
            return float(value)
        return value

    def total_length_to_contraction(self, total_length: np.ndarray | float) -> np.ndarray | float:
        """Return total contraction from the fully extended 40 mm reference [m]."""
        value = self.actuator_total_initial_length - np.asarray(total_length, dtype=np.float64)
        if np.isscalar(total_length):
            return float(value)
        return value

    def total_length_to_stroke_fraction(self, total_length: np.ndarray | float) -> np.ndarray | float:
        """Return normalized stroke where 0 is 40 mm and 1 is the 8 mm lower stop."""
        value = self.total_length_to_contraction(total_length) / self.actuator_max_contraction
        if np.isscalar(total_length):
            return float(value)
        return value

    def angle_to_total_length(
        self,
        angle: np.ndarray | float,
        angle_initial: float,
        initial_total_length: float | None = None,
        clip: bool = True,
    ) -> np.ndarray | float:
        """Convert encoder angle to total actuator length [m].

        ``initial_total_length`` is the unknown pump-on length at CSV start.
        Tuning can treat it as a free parameter while angle remains relative.
        """
        if initial_total_length is None:
            initial_total_length = self.init_total_length
        total_length = float(initial_total_length) + np.asarray(
            self.angle_to_length_delta(angle, angle_initial), dtype=np.float64
        )
        if clip:
            return self.clamp_total_length(total_length)
        if np.isscalar(angle):
            return float(total_length)
        return total_length

    def angle_to_cell_length(
        self,
        angle: np.ndarray | float,
        angle_initial: float,
        initial_total_length: float | None = None,
        clip: bool = True,
    ) -> np.ndarray | float:
        """Convert encoder angle to lib4's per-cell length coordinate [m]."""
        return self.total_length_to_cell_length(
            self.angle_to_total_length(angle, angle_initial, initial_total_length, clip)
        )

    def angular_velocity_to_linear_velocity(
        self,
        angular_velocity: np.ndarray | float,
    ) -> np.ndarray | float:
        """Convert encoder angular velocity to total relative actuator velocity [m/s]."""
        omega = np.asarray(angular_velocity, dtype=np.float64) * self.encoder_angular_velocity_scale
        velocity = (
            self.encoder_displacement_sign
            * self.encoder_pitch_radius
            * self.encoder_gear_ratio
            * self._angle_delta_to_rad(omega)
        )
        if np.isscalar(angular_velocity):
            return float(velocity)
        return velocity

    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325]),
    ) -> tuple[np.ndarray, dict]:
        ctrl = np.asarray(ctrl, dtype=np.float64).copy()
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if ctrl.shape != (6,):
            raise ValueError(f"sim4 expects 6 control inputs, got shape {ctrl.shape}")
        if goal.size < 2:
            raise ValueError(f"sim4 expects at least 2 actuator reference inputs, got shape {goal.shape}")
        goal_press = goal[:2]
        ref_displacement = float(goal[2]) if goal.size >= 3 else np.nan

        if self.is_pid:
            pid_delta = self.pid.get_action(obs=self.obs[2:4], ref=goal_press)
            ctrl[2:6] += pid_delta
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        ctrl = np.clip(ctrl, 0.0, 1.0)

        if self.is_anti_windup:
            self.pid.anti_windup(ctrl=original_ctrl[2:6], sat_ctrl=ctrl[2:6])

        if self.scale:
            ctrl = self.scale_low + (self.scale_high - self.scale_low) * ctrl

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
            sen_cell_length=next_obs[5],
            sen_cell_velocity=next_obs[6],
            sen_total_length=self.cell_length_to_total_length(next_obs[5]),
            sen_total_velocity=self.cell_to_total_displacement(next_obs[6]),
            sen_total_length_delta=self.cell_to_total_displacement(next_obs[5] - self.init_length),
            sen_total_contraction=self.cell_to_total_displacement(self.init_length - next_obs[5]),
            sen_total_contraction_from_rest=self.total_length_to_contraction(
                self.cell_length_to_total_length(next_obs[5])
            ),
            sen_total_stroke_fraction=self.total_length_to_stroke_fraction(
                self.cell_length_to_total_length(next_obs[5])
            ),
            actuator_total_initial_length=self.actuator_total_initial_length,
            actuator_min_total_length=self.actuator_min_total_length,
            actuator_max_contraction=self.actuator_max_contraction,
            actuator_hard_min_cell_length=self.actuator_hard_min_cell_length,
            contraction_ratio=self.lib.get_contraction_ratio(next_obs[5]),
            ref_pos=101.325,
            ref_neg=101.325,
            ref_act_pos=goal_press[0],
            ref_act_neg=goal_press[1],
            ref_displacement=ref_displacement,
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
        init_length = self.clamp_cell_length(init_length)
        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.init_act_pos_press = init_act_pos_press
        self.init_act_neg_press = init_act_neg_press
        self.init_length = init_length
        self.init_total_length = self.cell_length_to_total_length(init_length)
        self.init_velocity = init_velocity
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

    def set_offset(
        self,
        pos_offset: float,
        neg_offset: float,
        act_pos_offset: float = 0,
        act_neg_offset: float = 0,
    ) -> None:
        self.offset_pos = pos_offset
        self.offset_neg = neg_offset
        self.offset_act_pos = act_pos_offset
        self.offset_act_neg = act_neg_offset

    def set_leak_coefficients(
        self,
        pos_atm: float = 0.0,
        neg_atm: float = 0.0,
        cross: float = 0.0,
    ) -> None:
        self.lib.set_leak_coefficients_c(
            max(0.0, float(pos_atm)),
            max(0.0, float(neg_atm)),
            max(0.0, float(cross)),
        )

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
