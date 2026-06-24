from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np
import os

from pneu_env.pid import ActuatorPressurePID
from pneu_env.real.real_act import PneuRealAct
from pneu_env.sim4 import (
    DEFAULT_ACTUATOR_MAX_CONTRACTION_M,
    DEFAULT_ACTUATOR_SERIES_COUNT,
    DEFAULT_ENCODER_DISPLACEMENT_SIGN,
    DEFAULT_ENCODER_GEAR_RATIO,
    DEFAULT_ENCODER_PITCH_RADIUS_M,
)


STD_RHO = 1.20411831637462


def _lpm_to_mass_flow(flow_lpm: float) -> float:
    return float(flow_lpm) * STD_RHO / 60000.0


class PneuReal:
    """Real lib4 wrapper.

    The real sensor provides absolute encoder angle, but env4 trains on
    displacement. This wrapper captures a per-run encoder zero and exposes
    relative displacement in millimeters to env4 while sending absolute
    ``angle_reference`` back to the TCP bridge.
    """

    def __init__(
        self,
        freq: float = 50.0,
        scale: bool = True,
        init_pos_press: float = 101.325,
        init_neg_press: float = 101.325,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
        actuator_initial_length: float = 0.02,
        actuator_series_count: float = DEFAULT_ACTUATOR_SERIES_COUNT,
        actuator_max_contraction: float = DEFAULT_ACTUATOR_MAX_CONTRACTION_M,
        actuator_min_total_length: float | None = None,
        encoder_pitch_radius: float = DEFAULT_ENCODER_PITCH_RADIUS_M,
        encoder_gear_ratio: float = DEFAULT_ENCODER_GEAR_RATIO,
        encoder_displacement_sign: float = DEFAULT_ENCODER_DISPLACEMENT_SIGN,
        encoder_angle_unit: Literal["deg", "rad"] = "deg",
        encoder_angular_velocity_scale: float = 1.0,
        encoder_zero_angle: float | None = None,
        initial_displacement_mm: float = 0.0,
        auto_zero_encoder: bool = True,
        clamp_displacement_ref: bool = True,
        unwrap_encoder_delta: bool = True,
        wrap_angle_reference: bool = True,
        **_: Any,
    ):
        self.freq = float(freq)
        self.scale = bool(scale)
        self.backend = PneuRealAct(freq=self.freq, scale=self.scale)
        self._step_count = 0
        self._last_time = 0.0
        self._raw_last_time: float | None = None
        self._time_origin: float | None = None
        self._stale_time_count = 0
        self._prev_press_vec: np.ndarray | None = None
        self._stale_obs_count = 0
        self._max_stale_obs_steps = int(os.getenv("PNEU_REAL4_MAX_STALE", "300"))
        self._debug_every = int(os.getenv("PNEU_REAL4_DEBUG_EVERY", "0"))

        self.actuator_series_count = float(actuator_series_count)
        self.actuator_cell_initial_length = float(actuator_initial_length)
        self.actuator_total_initial_length = (
            self.actuator_cell_initial_length * self.actuator_series_count
        )
        if actuator_min_total_length is None:
            actuator_min_total_length = (
                self.actuator_total_initial_length - float(actuator_max_contraction)
            )
        self.actuator_min_total_length = float(actuator_min_total_length)
        self.actuator_max_contraction = (
            self.actuator_total_initial_length - self.actuator_min_total_length
        )
        self.initial_length = self.actuator_cell_initial_length
        self.minimum_length = max(0.0, self.actuator_min_total_length / self.actuator_series_count)
        self.init_length = self.initial_length
        self.init_total_length = self.cell_length_to_total_length(self.init_length)
        self.init_velocity = 0.0

        self.encoder_pitch_radius = float(encoder_pitch_radius)
        self.encoder_gear_ratio = float(encoder_gear_ratio)
        self.encoder_displacement_sign = float(encoder_displacement_sign)
        self.encoder_angle_unit = str(encoder_angle_unit)
        self.encoder_angular_velocity_scale = float(encoder_angular_velocity_scale)
        self.auto_zero_encoder = bool(auto_zero_encoder)
        self.unwrap_encoder_delta = bool(unwrap_encoder_delta)
        self.wrap_angle_reference = bool(wrap_angle_reference)
        self.clamp_displacement_ref = bool(clamp_displacement_ref)
        self._angle_zero = None if encoder_zero_angle is None else float(encoder_zero_angle)
        self._encoder_zero_locked = encoder_zero_angle is not None
        self.initial_displacement_m = self._clamp_displacement(float(initial_displacement_mm) * 1e-3)

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

        self.is_pid = False
        self.is_anti_windup = False
        self.obs = np.array(
            [
                init_pos_press,
                init_neg_press,
                init_act_pos_press,
                init_act_neg_press,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.set_init_press(
            init_pos_press,
            init_neg_press,
            init_act_pos_press,
            init_act_neg_press,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    @property
    def angle_zero(self) -> float | None:
        return self._angle_zero

    def _angle_delta_to_rad(self, angle_delta: np.ndarray | float) -> np.ndarray | float:
        if self.encoder_angle_unit == "deg":
            return np.deg2rad(angle_delta)
        return angle_delta

    def _rad_to_angle_delta(self, rad_delta: np.ndarray | float) -> np.ndarray | float:
        if self.encoder_angle_unit == "deg":
            return np.rad2deg(rad_delta)
        return rad_delta

    def _wrap_delta(self, delta: float) -> float:
        if not self.unwrap_encoder_delta:
            return float(delta)
        if self.encoder_angle_unit == "deg":
            return float((delta + 180.0) % 360.0 - 180.0)
        return float((delta + np.pi) % (2.0 * np.pi) - np.pi)

    def _wrap_angle(self, angle: float) -> float:
        if not self.wrap_angle_reference:
            return float(angle)
        if self.encoder_angle_unit == "deg":
            return float(angle % 360.0)
        return float(angle % (2.0 * np.pi))

    def angle_to_displacement(
        self,
        angle: np.ndarray | float,
        angle_initial: float | None = None,
    ) -> np.ndarray | float:
        if angle_initial is None:
            angle_initial = self._ensure_encoder_zero()
        delta = np.asarray(angle, dtype=np.float64) - float(angle_initial)
        if np.isscalar(angle):
            delta = self._wrap_delta(float(delta))
        elif self.unwrap_encoder_delta:
            if self.encoder_angle_unit == "deg":
                delta = (delta + 180.0) % 360.0 - 180.0
            else:
                delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        displacement = (
            self.encoder_displacement_sign
            * self.encoder_pitch_radius
            * self.encoder_gear_ratio
            * self._angle_delta_to_rad(delta)
        )
        if np.isscalar(angle):
            return float(displacement)
        return displacement

    def angular_velocity_to_linear_velocity(
        self,
        angular_velocity: np.ndarray | float,
    ) -> np.ndarray | float:
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

    def displacement_to_angle(
        self,
        displacement_m: float,
        angle_initial: float | None = None,
    ) -> float:
        if angle_initial is None:
            angle_initial = self._ensure_encoder_zero()
        denom = self.encoder_displacement_sign * self.encoder_pitch_radius * self.encoder_gear_ratio
        relative_displacement_m = float(displacement_m) - self.initial_displacement_m
        angle_delta_rad = relative_displacement_m / denom
        return self._wrap_angle(float(angle_initial) + float(self._rad_to_angle_delta(angle_delta_rad)))

    def _clamp_displacement(self, displacement_m: float) -> float:
        return float(np.clip(displacement_m, 0.0, self.actuator_max_contraction))

    def cell_to_total_displacement(self, cell_displacement: np.ndarray | float) -> np.ndarray | float:
        value = np.asarray(cell_displacement, dtype=np.float64) * self.actuator_series_count
        if np.isscalar(cell_displacement):
            return float(value)
        return value

    def total_to_cell_displacement(self, total_displacement: np.ndarray | float) -> np.ndarray | float:
        value = np.asarray(total_displacement, dtype=np.float64) / self.actuator_series_count
        if np.isscalar(total_displacement):
            return float(value)
        return value

    def cell_length_to_total_length(self, cell_length: np.ndarray | float) -> np.ndarray | float:
        return self.cell_to_total_displacement(cell_length)

    def total_length_to_cell_length(self, total_length: np.ndarray | float) -> np.ndarray | float:
        return self.total_to_cell_displacement(total_length)

    def clamp_cell_length(self, cell_length: np.ndarray | float) -> np.ndarray | float:
        value = np.clip(np.asarray(cell_length, dtype=np.float64), self.minimum_length, self.initial_length)
        if np.isscalar(cell_length):
            return float(value)
        return value

    def total_length_to_contraction(self, total_length: np.ndarray | float) -> np.ndarray | float:
        value = self.actuator_total_initial_length - np.asarray(total_length, dtype=np.float64)
        if np.isscalar(total_length):
            return float(value)
        return value

    def total_length_to_stroke_fraction(self, total_length: np.ndarray | float) -> np.ndarray | float:
        value = self.total_length_to_contraction(total_length) / self.actuator_max_contraction
        if np.isscalar(total_length):
            return float(value)
        return value

    def _ensure_encoder_zero(self) -> float:
        if self._angle_zero is not None:
            return self._angle_zero
        if self.auto_zero_encoder:
            self.backend.read_obs_file()
        angle = float(getattr(self.backend, "angle", 0.0))
        if not np.isfinite(angle):
            angle = 0.0
        self._angle_zero = angle
        self.backend.angle_reference = angle
        print(f"[ INFO] real4 encoder zero: {self._angle_zero:.6f} {self.encoder_angle_unit}")
        return self._angle_zero

    def reset_encoder_zero(self, angle: float | None = None) -> float:
        if angle is None:
            self.backend.read_obs_file()
            angle = float(getattr(self.backend, "angle", 0.0))
        if not np.isfinite(angle):
            raise ValueError(f"invalid encoder zero angle: {angle}")
        self._angle_zero = float(angle)
        self.backend.angle_reference = float(angle)
        return self._angle_zero

    def _flowrate_lpm_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.backend.flowrate1),
            float(self.backend.flowrate2),
            float(self.backend.flowrate3),
            float(self.backend.flowrate4),
            float(self.backend.flowrate5),
            float(self.backend.flowrate6),
        )

    def _split_ctrl(self, ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(-1).copy()
        if ctrl.size == 2:
            ctrl = np.array([ctrl[0], ctrl[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif ctrl.size != 6:
            raise ValueError(f"ctrl must be length 2 or 6, got {ctrl.size}")
        if not np.all(np.isfinite(ctrl)):
            ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=1.0, neginf=0.0)
        ctrl = np.clip(ctrl, 0.0, 1.0)
        return ctrl[:2], ctrl[2:]

    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325, 0.0], dtype=np.float64),
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        self._step_count += 1
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(goal)):
            goal = np.nan_to_num(goal, nan=101.325, posinf=101.325, neginf=101.325)
        goal_posneg = goal[:2] if goal.size >= 2 else np.array([101.325, 101.325], dtype=np.float64)
        ref_displacement_mm = float(goal[2]) if goal.size >= 3 else self.initial_displacement_m * 1e3
        ref_displacement_m = ref_displacement_mm * 1e-3
        if self.clamp_displacement_ref:
            ref_displacement_m = self._clamp_displacement(ref_displacement_m)
            ref_displacement_mm = ref_displacement_m * 1e3
        commanded_angle_ref = self.displacement_to_angle(ref_displacement_m)
        self.backend.angle_reference = commanded_angle_ref

        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(-1).copy()
        if ctrl.size == 2:
            ctrl = np.array([ctrl[0], ctrl[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif ctrl.size != 6:
            raise ValueError(f"ctrl must be length 2 or 6, got {ctrl.size}")
        if not np.all(np.isfinite(ctrl)):
            ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=1.0, neginf=0.0)

        if self.is_pid:
            pid_delta = self.pid.get_action(obs=self.obs[2:4], ref=goal_posneg)
            ctrl[2:6] += pid_delta
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        ctrl = np.clip(ctrl, 0.0, 1.0)
        if self.is_anti_windup:
            self.pid.anti_windup(
                ctrl=original_ctrl[2:6],
                sat_ctrl=ctrl[2:6],
            )

        main_ctrl, act_ctrls = self._split_ctrl(ctrl)
        ctrl_unit = np.r_[main_ctrl, act_ctrls]
        if self.scale:
            backend_ctrl = 1.5 * ctrl_unit - 0.5
        else:
            backend_ctrl = 2.0 * ctrl_unit - 1.0
        backend_ctrl = np.clip(backend_ctrl, -1.0, 1.0)

        _, raw_info = self.backend.observe(ctrl=backend_ctrl, goal=goal_posneg)
        obs = raw_info["Observation"]

        dt = 1.0 / max(self.freq, 1e-6)
        raw_time = float(obs.get("time", np.nan))
        is_stale_or_invalid = (not np.isfinite(raw_time)) or (
            self._raw_last_time is not None and raw_time <= self._raw_last_time
        )
        if is_stale_or_invalid:
            self._stale_time_count += 1
            prev_raw_time = self._raw_last_time if self._raw_last_time is not None else 0.0
            raw_time = prev_raw_time + dt
            if self._stale_time_count in (50, 200, 500):
                print("[WARN] real4 observe time is stale. Using local fallback time.")
        else:
            self._stale_time_count = 0
        self._raw_last_time = raw_time
        if self._time_origin is None:
            self._time_origin = raw_time
        time_value = raw_time - self._time_origin
        if time_value < self._last_time:
            time_value = self._last_time + dt
        self._last_time = time_value

        angle = float(obs.get("angle", getattr(self.backend, "angle", 0.0)))
        angular_vel = float(obs.get("angular_vel", getattr(self.backend, "angular_vel", 0.0)))
        relative_displacement_m = self.angle_to_displacement(angle)
        displacement_m = self._clamp_displacement(self.initial_displacement_m + relative_displacement_m)
        displacement_vel_m_s = self.angular_velocity_to_linear_velocity(angular_vel)
        total_length = self.actuator_total_initial_length - displacement_m
        cell_length = self.clamp_cell_length(self.total_length_to_cell_length(total_length))
        cell_velocity = -self.total_to_cell_displacement(displacement_vel_m_s)

        next_obs = np.array(
            [
                time_value,
                float(obs.get("pos_press", np.nan)),
                float(obs.get("neg_press", np.nan)),
                float(obs.get("act_pos_press", np.nan)),
                float(obs.get("act_neg_press", np.nan)),
                cell_length,
                cell_velocity,
            ],
            dtype=np.float64,
        )
        self.obs = next_obs[1:7].copy()

        press_vec = next_obs[1:5].copy()
        if np.all(np.isfinite(press_vec)):
            if self._prev_press_vec is not None and np.allclose(press_vec, self._prev_press_vec, rtol=0.0, atol=1e-9):
                self._stale_obs_count += 1
                if self._stale_obs_count in (100, 500):
                    print(
                        "[WARN] real4 pressure observation is unchanged for "
                        f"{self._stale_obs_count} steps."
                    )
                if self._max_stale_obs_steps > 0 and self._stale_obs_count >= self._max_stale_obs_steps:
                    raise RuntimeError(
                        "real4 observation stream appears stalled "
                        f"({self._stale_obs_count} unchanged steps)."
                    )
            else:
                self._stale_obs_count = 0
            self._prev_press_vec = press_vec

        if self._debug_every > 0 and self._step_count % self._debug_every == 0:
            print(
                "[DBG] real4 "
                f"step={self._step_count} time={next_obs[0]:.3f} "
                f"angle={angle:.3f} zero={self._ensure_encoder_zero():.3f} "
                f"disp_mm={displacement_m * 1e3:.3f} ref_mm={ref_displacement_mm:.3f} "
                f"angle_ref={commanded_angle_ref:.3f}"
            )

        flowrate1_lpm, flowrate2_lpm, flowrate3_lpm, flowrate4_lpm, flowrate5_lpm, flowrate6_lpm = (
            self._flowrate_lpm_tuple()
        )
        sen_total_length = self.cell_length_to_total_length(cell_length)
        sen_total_velocity = self.cell_to_total_displacement(cell_velocity)
        observation_info = dict(
            curr_time=next_obs[0],
            sen_pos=next_obs[1],
            sen_neg=next_obs[2],
            sen_act_pos=next_obs[3],
            sen_act_neg=next_obs[4],
            P1_pos=next_obs[3],
            P1_neg=next_obs[4],
            sen_length=cell_length,
            sen_velocity=cell_velocity,
            sen_cell_length=cell_length,
            sen_cell_velocity=cell_velocity,
            sen_total_length=sen_total_length,
            sen_total_velocity=sen_total_velocity,
            sen_total_contraction=displacement_m,
            sen_total_contraction_from_rest=displacement_m,
            sen_relative_displacement=relative_displacement_m,
            sen_total_stroke_fraction=self.total_length_to_stroke_fraction(sen_total_length),
            sen_env_displacement=displacement_m * 1e3,
            sen_env_velocity=displacement_vel_m_s * 1e3,
            ref_pos=float(goal_posneg[0]),
            ref_neg=float(goal_posneg[1]),
            ref_act_pos=float(obs.get("act_pos_ref", goal_posneg[0])),
            ref_act_neg=float(obs.get("act_neg_ref", goal_posneg[1])),
            ref_displacement=ref_displacement_mm,
            ref_displacement_raw=float(goal[2]) if goal.size >= 3 else self.initial_displacement_m * 1e3,
            angle=angle,
            angle_zero=self._ensure_encoder_zero(),
            angle_reference=commanded_angle_ref,
            angular_vel=angular_vel,
            ctrl_pos=float(obs.get("pos_ctrl", np.nan)),
            ctrl_neg=float(obs.get("neg_ctrl", np.nan)),
            ctrl_act_pos_in=float(obs.get("act_pos_ctrl1", np.nan)),
            ctrl_act_pos_out=float(obs.get("act_pos_ctrl2", np.nan)),
            ctrl_act_neg_in=float(obs.get("act_neg_ctrl1", np.nan)),
            ctrl_act_neg_out=float(obs.get("act_neg_ctrl2", np.nan)),
            act_pos_ctrl1=float(obs.get("act_pos_ctrl1", np.nan)),
            act_pos_ctrl2=float(obs.get("act_pos_ctrl2", np.nan)),
            act_neg_ctrl1=float(obs.get("act_neg_ctrl1", np.nan)),
            act_neg_ctrl2=float(obs.get("act_neg_ctrl2", np.nan)),
            flowrate1=flowrate1_lpm,
            flowrate2=flowrate2_lpm,
            flowrate3=flowrate3_lpm,
            flowrate4=flowrate4_lpm,
            flowrate5=flowrate5_lpm,
            flowrate6=flowrate6_lpm,
        )
        return next_obs, {"obs_w/o_noise": next_obs.copy(), "Observation": observation_info}

    def get_mass_flowrate_dict(self) -> Dict[str, float]:
        flowrate1_lpm, flowrate2_lpm, flowrate3_lpm, flowrate4_lpm, flowrate5_lpm, flowrate6_lpm = (
            self._flowrate_lpm_tuple()
        )
        flow1 = _lpm_to_mass_flow(flowrate1_lpm)
        flow2 = _lpm_to_mass_flow(flowrate2_lpm)
        flow3 = _lpm_to_mass_flow(flowrate3_lpm)
        flow4 = _lpm_to_mass_flow(flowrate4_lpm)
        flow5 = _lpm_to_mass_flow(flowrate5_lpm)
        flow6 = _lpm_to_mass_flow(flowrate6_lpm)
        return dict(
            flow1=flow1,
            flow2=flow2,
            flow3=flow3,
            flow4=flow4,
            flow5=flow5,
            flow6=flow6,
            chamber_pos_valve=flow1,
            chamber_neg_valve=flow2,
            act_pos_in=flow3,
            act_pos_out=flow4,
            act_neg_in=flow5,
            act_neg_out=flow6,
            act_pos_net_in=flow3 - flow4,
            act_neg_net_in=flow5 - flow6,
            flowrate1_lpm=flowrate1_lpm,
            flowrate2_lpm=flowrate2_lpm,
            flowrate3_lpm=flowrate3_lpm,
            flowrate4_lpm=flowrate4_lpm,
            flowrate5_lpm=flowrate5_lpm,
            flowrate6_lpm=flowrate6_lpm,
        )

    def get_mean_mass_flowrate(self) -> Dict[str, float]:
        return self.get_mass_flowrate_dict()

    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
        init_length: float | None = None,
        init_velocity: float = 0.0,
    ) -> None:
        self._last_time = 0.0
        self._raw_last_time = None
        self._time_origin = None
        self._stale_time_count = 0
        self._stale_obs_count = 0
        self._prev_press_vec = None
        if self.auto_zero_encoder and not self._encoder_zero_locked:
            self._angle_zero = None

        self.backend.pos_press = float(init_pos_press)
        self.backend.neg_press = float(init_neg_press)
        self.backend.act_pos_press = float(init_act_pos_press)
        self.backend.act_neg_press = float(init_act_neg_press)
        self.backend.obs = np.array(
            [self.backend.act_pos_press, self.backend.act_neg_press],
            dtype=np.float32,
        )
        if init_length is not None:
            self.init_length = self.clamp_cell_length(init_length)
            self.init_total_length = self.cell_length_to_total_length(self.init_length)
        self.init_velocity = float(init_velocity)
        self.obs = np.array(
            [
                self.backend.pos_press,
                self.backend.neg_press,
                self.backend.act_pos_press,
                self.backend.act_neg_press,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    def set_offset(
        self,
        pos_offset: float,
        neg_offset: float,
        act_pos_offset: float = 0.0,
        act_neg_offset: float = 0.0,
    ) -> None:
        del pos_offset, neg_offset, act_pos_offset, act_neg_offset

    def set_volume(self, vol1: float, vol2: float) -> None:
        del vol1, vol2

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
