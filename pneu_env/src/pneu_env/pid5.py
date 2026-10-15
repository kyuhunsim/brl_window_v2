from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PressureLoop5:
    kp: float = 0.03
    ki: float = 0.0
    kd: float = 0.0
    deadband: float = 0.5
    integral_limit: float = 100.0
    output_limit: float = 1.0


class PneumaticController5:
    """Four pressure loops with mutually exclusive six-valve allocation.

    Reference order is [ch_pos, ch_neg, act_pos, act_neg, displacement].
    The controller emits normalized commands in the same order as lib4.
    """

    def __init__(
        self,
        freq: float = 50.0,
        chamber_pos: dict[str, float] | None = None,
        chamber_neg: dict[str, float] | None = None,
        actuator_pos: dict[str, float] | None = None,
        actuator_neg: dict[str, float] | None = None,
        closed_signal: float = 0.0,
        min_open: float = 0.0,
        max_open: float = 1.0,
        valve_gamma: float = 1.0,
        reserve_freeze_threshold: float = 2.0,
    ):
        self.dt = 1.0 / float(freq)
        self.loops = (
            PressureLoop5(**(chamber_pos or {})),
            PressureLoop5(**(chamber_neg or {})),
            PressureLoop5(**(actuator_pos or {})),
            PressureLoop5(**(actuator_neg or {})),
        )
        self.closed_signal = float(closed_signal)
        self.min_open = float(min_open)
        self.max_open = float(max_open)
        self.valve_gamma = float(valve_gamma)
        self.reserve_freeze_threshold = float(reserve_freeze_threshold)
        self.integral = np.zeros(4, dtype=np.float64)
        self.prev_error = np.zeros(4, dtype=np.float64)
        self.last_info: dict[str, Any] = {}

    def reset(self) -> None:
        self.integral.fill(0.0)
        self.prev_error.fill(0.0)
        self.last_info = {}

    def _effort(self, index: int, error: float, integrate: bool = True) -> float:
        cfg = self.loops[index]
        active_error = 0.0 if abs(error) <= cfg.deadband else float(error)
        if integrate:
            self.integral[index] = np.clip(
                self.integral[index] + active_error * self.dt,
                -cfg.integral_limit,
                cfg.integral_limit,
            )
        derivative = (active_error - self.prev_error[index]) / max(self.dt, 1e-9)
        self.prev_error[index] = active_error
        value = cfg.kp * active_error + cfg.ki * self.integral[index] + cfg.kd * derivative
        return float(np.clip(value, -cfg.output_limit, cfg.output_limit))

    def _open(self, magnitude: float) -> float:
        if magnitude <= 0.0:
            return self.closed_signal
        normalized = np.clip(magnitude, 0.0, 1.0) ** (1.0/max(self.valve_gamma, 1e-6))
        command = self.min_open + (self.max_open-self.min_open)*normalized
        return float(np.clip(command, self.closed_signal, self.max_open))

    def get_action(self, obs: np.ndarray, ref: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        if obs.size < 4 or ref.size < 4:
            raise ValueError("controller5 needs four pressures and four pressure references")

        ch_pos, ch_neg, act_pos, act_neg = obs[:4]
        ref_ch_pos, ref_ch_neg, ref_act_pos, ref_act_neg = ref[:4]

        # ctrl1 vents positive chamber: more opening lowers pressure.
        ch_pos_effort = self._effort(0, ch_pos - ref_ch_pos)
        # ctrl2 evacuates negative chamber: more opening lowers pressure.
        ch_neg_effort = self._effort(1, ch_neg - ref_ch_neg)

        pos_reserve = ch_pos - act_pos
        neg_reserve = act_neg - ch_neg
        pos_error = ref_act_pos - act_pos
        neg_error = ref_act_neg - act_neg
        pos_freeze = pos_error > 0.0 and pos_reserve < self.reserve_freeze_threshold
        neg_freeze = neg_error < 0.0 and neg_reserve < self.reserve_freeze_threshold
        act_pos_effort = self._effort(2, pos_error, integrate=not pos_freeze)
        act_neg_effort = self._effort(3, neg_error, integrate=not neg_freeze)

        ctrl = np.full(6, self.closed_signal, dtype=np.float64)
        ctrl[0] = self._open(max(ch_pos_effort, 0.0))
        ctrl[1] = self._open(max(ch_neg_effort, 0.0))
        if act_pos_effort > 0.0 and not pos_freeze:
            ctrl[2] = self._open(act_pos_effort)
        elif act_pos_effort < 0.0:
            ctrl[3] = self._open(-act_pos_effort)
        if act_neg_effort > 0.0:
            ctrl[4] = self._open(act_neg_effort)
        elif act_neg_effort < 0.0 and not neg_freeze:
            ctrl[5] = self._open(-act_neg_effort)

        self.last_info = dict(
            pressure_error=np.array([ch_pos-ref_ch_pos, ch_neg-ref_ch_neg, pos_error, neg_error]),
            reserve=np.array([pos_reserve, neg_reserve]),
            integrator_frozen=np.array([False, False, pos_freeze, neg_freeze]),
            base_ctrl=ctrl.copy(),
        )
        return ctrl

    def apply_residual(
        self,
        base_ctrl: np.ndarray,
        residual: np.ndarray,
        valve_residual_limit: float = 0.05,
    ) -> np.ndarray:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        ctrl = np.asarray(base_ctrl, dtype=np.float64).copy()
        if residual.size >= 8:
            ctrl += np.clip(residual[2:8], -valve_residual_limit, valve_residual_limit)

        # Preserve directional allocation even after the learned correction.
        if ctrl[2] >= ctrl[3]:
            ctrl[3] = self.closed_signal
        else:
            ctrl[2] = self.closed_signal
        if ctrl[4] >= ctrl[5]:
            ctrl[5] = self.closed_signal
        else:
            ctrl[4] = self.closed_signal
        return np.clip(ctrl, 0.0, 1.0)
