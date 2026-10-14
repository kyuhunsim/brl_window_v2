"""Constant two-channel pressure reference."""

from __future__ import annotations


class ConstantPressureRef:
    def __init__(self, pos: float, neg: float):
        self.pos = float(pos)
        self.neg = float(neg)

    def time_reset(self) -> None:
        return None

    def get_goal(self, curr_time: float) -> tuple[float, float]:
        del curr_time
        return self.pos, self.neg
