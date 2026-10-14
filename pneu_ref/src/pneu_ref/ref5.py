from __future__ import annotations

from typing import Any

import numpy as np

from pneu_ref.base_ref import BaseRef


class FlowAwarePressureDisplacementRef(BaseRef):
    """Extend [act_pos, act_neg, displacement] with chamber reserve targets."""

    def __init__(
        self,
        actuator_displacement_ref: Any,
        base_margin_pos: float = 25.0,
        base_margin_neg: float = 35.0,
        rate_margin_pos: float = 0.10,
        rate_margin_neg: float = 0.10,
        derivative_dt: float = 0.02,
        actuator_volume_pos_m3: float = 5.0e-5,
        actuator_volume_neg_m3: float = 5.0e-5,
        gas_constant: float = 287.05,
        gas_temperature_k: float = 293.15,
        valve_flow_coeff: float = 5.0e-7,
        flow_safety_factor: float = 1.25,
        ch_pos_min: float = 110.0,
        ch_pos_max: float = 500.0,
        ch_neg_min: float = 5.0,
        ch_neg_max: float = 90.0,
    ):
        super().__init__()
        self.base_ref = actuator_displacement_ref
        self.base_margin_pos = float(base_margin_pos)
        self.base_margin_neg = float(base_margin_neg)
        self.rate_margin_pos = float(rate_margin_pos)
        self.rate_margin_neg = float(rate_margin_neg)
        self.derivative_dt = float(derivative_dt)
        self.actuator_volume = np.array(
            [actuator_volume_pos_m3, actuator_volume_neg_m3], dtype=np.float64
        )
        self.gas_constant = float(gas_constant)
        self.gas_temperature_k = float(gas_temperature_k)
        self.valve_flow_coeff = float(valve_flow_coeff)
        self.flow_safety_factor = float(flow_safety_factor)
        self.ch_pos_min = float(ch_pos_min)
        self.ch_pos_max = float(ch_pos_max)
        self.ch_neg_min = float(ch_neg_min)
        self.ch_neg_max = float(ch_neg_max)
        self.max_time = getattr(actuator_displacement_ref, "max_time", float("inf"))
        self._prev_time: float | None = None
        self._prev_act = np.zeros(2, dtype=np.float64)

    def time_reset(self) -> None:
        if hasattr(self.base_ref, "time_reset"):
            self.base_ref.time_reset()
        self._prev_time = None
        self._prev_act.fill(0.0)

    def get_goal(self, curr_time: float) -> tuple[float, float, float, float, float]:
        act_pos, act_neg, displacement = self.base_ref.get_goal(curr_time)
        current_act = np.array([act_pos, act_neg], dtype=np.float64)
        if self._prev_time is None or curr_time <= self._prev_time:
            act_rate = np.zeros(2, dtype=np.float64)
        else:
            dt = max(float(curr_time) - self._prev_time, self.derivative_dt)
            act_rate = (current_act - self._prev_act) / dt
        self._prev_time = float(curr_time)
        self._prev_act = current_act
        act_pos_rate, act_neg_rate = act_rate

        # Isothermal pressure dynamics give m_dot ~= V/(R*T) * dP/dt.
        required_rate = np.array([max(act_pos_rate, 0.0), max(-act_neg_rate, 0.0)])
        required_mass_flow = (
            self.actuator_volume
            / (self.gas_constant*self.gas_temperature_k)
            * required_rate*1000.0
            * self.flow_safety_factor
        )
        # Invert the low-order valve relation m_dot = C*sqrt(delta_P).
        flow_delta_p_kpa = (
            required_mass_flow/max(self.valve_flow_coeff, 1e-12)
        ) ** 2 / 1000.0
        margin_pos = max(
            self.base_margin_pos + self.rate_margin_pos*required_rate[0],
            self.base_margin_pos + flow_delta_p_kpa[0],
        )
        margin_neg = max(
            self.base_margin_neg + self.rate_margin_neg*required_rate[1],
            self.base_margin_neg + flow_delta_p_kpa[1],
        )
        ch_pos = np.clip(float(act_pos) + margin_pos, self.ch_pos_min, self.ch_pos_max)
        ch_neg = np.clip(float(act_neg) - margin_neg, self.ch_neg_min, self.ch_neg_max)
        return ch_pos, ch_neg, float(act_pos), float(act_neg), float(displacement)
