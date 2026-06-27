from __future__ import annotations

import math
import random
from typing import Any

from pneu_ref.base_ref import BaseRef


class RandomDispRef(BaseRef):
    """Random scalar displacement reference in millimeters."""

    def __init__(
        self,
        max_ts: int = 5,
        max_amp: float = 4.0,
        max_per: float = 10.0,
        min_off: float = 4.0,
        max_off: float = 24.0,
        seed: int = 61099,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.max_ts = int(max_ts)
        self.max_amp = float(max_amp)
        self.max_per = float(max_per)
        self.min_off = float(min_off)
        self.max_off = float(max_off)
        self.start_time = 0.0
        self.time_step = 0.0
        self.amplitude = 0.0
        self.period = 1.0
        self.offset = self.min_off

    def time_reset(self) -> None:
        self.start_time = 0.0

    def get_goal(self, curr_time: float) -> float:
        dt = curr_time - self.start_time
        if dt > self.time_step:
            self.start_time = curr_time
            self.time_step = self.rng.randrange(1, self.max_ts + 1)
            self.amplitude = self.rng.uniform(-self.max_amp, self.max_amp)
            self.period = 0.1 * self.rng.randrange(20, int(10 * self.max_per) + 1)
            self.offset = self.rng.uniform(self.min_off, self.max_off)
        return self.amplitude * math.sin(2 * math.pi * dt / self.period) + self.offset


class PressureDisplacementRef(BaseRef):
    """Combine an existing pressure-pair ref with a displacement scalar ref."""

    def __init__(
        self,
        pressure_ref: Any,
        displacement_ref: Any,
    ):
        super().__init__()
        self.pressure_ref = pressure_ref
        self.displacement_ref = displacement_ref
        self.max_time = min(
            getattr(pressure_ref, "max_time", float("inf")),
            getattr(displacement_ref, "max_time", float("inf")),
        )

    def time_reset(self) -> None:
        if hasattr(self.pressure_ref, "time_reset"):
            self.pressure_ref.time_reset()
        if hasattr(self.displacement_ref, "time_reset"):
            self.displacement_ref.time_reset()

    def get_goal(self, curr_time: float) -> tuple[float, float, float]:
        pos_ref, neg_ref = self.pressure_ref.get_goal(curr_time)
        disp_ref = self.displacement_ref.get_goal(curr_time)
        return float(pos_ref), float(neg_ref), float(disp_ref)


class PressureDiffDisplacementRef(BaseRef):
    """Pressure-pair ref with displacement constrained by pressure difference.

    The generated target keeps ``disp_ref ~= g(pos_ref - neg_ref)`` so the
    pressure and motion objectives do not ask the soft actuator for conflicting
    states.
    """

    def __init__(
        self,
        pressure_ref: Any,
        dp_to_disp_slope: float = 0.6011927721229401,
        dp_to_disp_intercept: float = -5.325254604545463,
        min_disp: float | None = None,
        max_disp: float | None = None,
    ):
        super().__init__()
        self.pressure_ref = pressure_ref
        self.dp_to_disp_slope = float(dp_to_disp_slope)
        self.dp_to_disp_intercept = float(dp_to_disp_intercept)
        self.min_disp = None if min_disp is None else float(min_disp)
        self.max_disp = None if max_disp is None else float(max_disp)
        self.max_time = getattr(pressure_ref, "max_time", float("inf"))

    def time_reset(self) -> None:
        if hasattr(self.pressure_ref, "time_reset"):
            self.pressure_ref.time_reset()

    def get_goal(self, curr_time: float) -> tuple[float, float, float]:
        pos_ref, neg_ref = self.pressure_ref.get_goal(curr_time)
        dp_ref = float(pos_ref) - float(neg_ref)
        disp_ref = self.dp_to_disp_slope * dp_ref + self.dp_to_disp_intercept
        if self.min_disp is not None:
            disp_ref = max(disp_ref, self.min_disp)
        if self.max_disp is not None:
            disp_ref = min(disp_ref, self.max_disp)
        return float(pos_ref), float(neg_ref), float(disp_ref)
