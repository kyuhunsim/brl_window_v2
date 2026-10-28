"""A fixed hold followed by one unchanged pressure-bank member for evaluation.

The initial hold is a *policy-controlled* segment, distinct from an
environment's pump-on/all-open preparation.  It gives the existing reference
history/preview buffer time to fill with consistent constant targets before
the bank begins.  This wrapper neither rebases pressure targets nor changes
the production reference-buffer implementation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pneu_ref.base_ref import BaseRef
from pneu_ref.optimized_trajectory_ref import OptimizedTrajectoryBankRef


class HoldThenBankRef(BaseRef):
    """Hold bank ``r(0)`` then replay ``r(t)`` once, clamping all later queries.

    ``get_goal`` is stateless in time, so a future preview cannot advance the
    current target.  Once ``hold_seconds + bank_seconds`` is reached, both
    current and preview queries return exactly the selected bank endpoint.
    This endpoint clamp is part of this evaluation protocol; it does not
    consume the unused tail of the source bank.
    """

    def __init__(
        self,
        *,
        trajectory_npz: str | Path,
        trajectory_index: int = 0,
        hold_seconds: float = 20.0,
        bank_seconds: float = 40.0,
    ) -> None:
        super().__init__()
        self.hold_seconds = float(hold_seconds)
        self.bank_seconds = float(bank_seconds)
        if not np.isfinite(self.hold_seconds) or self.hold_seconds < 0:
            raise ValueError("hold_seconds must be finite and nonnegative")
        if not np.isfinite(self.bank_seconds) or self.bank_seconds <= 0:
            raise ValueError("bank_seconds must be finite and positive")
        if isinstance(trajectory_index, bool) or int(trajectory_index) != trajectory_index:
            raise ValueError("trajectory_index must be an integer")
        self.trajectory_index = int(trajectory_index)
        self.trajectory_npz = Path(trajectory_npz).resolve()
        self._bank_sha256 = hashlib.sha256(self.trajectory_npz.read_bytes()).hexdigest()
        self.bank = OptimizedTrajectoryBankRef(
            trajectory_npz=str(self.trajectory_npz),
            trajectory_index=self.trajectory_index,
            repeat_after_end=False,
            repeat_same=False,
        )
        member = self.bank.trajectories[self.trajectory_index]
        if not np.isclose(float(member["time"][0]), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("selected bank member must start at time zero")
        self.source_duration_seconds = float(member["duration"])
        if self.bank_seconds > self.source_duration_seconds + 1e-12:
            raise ValueError("bank_seconds exceeds the selected bank member duration")
        self.hold_target_kpa = self.bank.get_goal(0.0)
        self.max_time = self.hold_seconds + self.bank_seconds
        if not np.isfinite(self.max_time):
            raise ValueError("total evaluation duration must be finite")

    def time_reset(self) -> None:
        """Select the same member again; preserve original target pressures."""
        self.bank.time_reset()

    def get_goal(self, curr_time: float) -> tuple[float, float]:
        t = float(curr_time)
        if not np.isfinite(t):
            raise ValueError("reference query time must be finite")
        bank_time = min(max(t - self.hold_seconds, 0.0), self.bank_seconds)
        return self.bank.get_goal(bank_time)

    def get_phase(self, curr_time: float) -> str:
        t = float(curr_time)
        if not np.isfinite(t):
            raise ValueError("reference query time must be finite")
        if t < self.hold_seconds:
            return "hold"
        if t < self.max_time:
            return "bank"
        return "end"

    def metadata(self) -> dict[str, Any]:
        """JSON-safe description of exactly the requested target trajectory."""
        return {
            "kind": "hold_then_fixed_bank",
            "trajectory_npz": str(self.trajectory_npz),
            "trajectory_npz_sha256": self._bank_sha256,
            "trajectory_index": self.trajectory_index,
            "hold_seconds": self.hold_seconds,
            "bank_seconds": self.bank_seconds,
            "duration_seconds": self.max_time,
            "source_duration_seconds": self.source_duration_seconds,
            "hold_target_kpa": list(self.hold_target_kpa),
            "bank_end_target_kpa": list(self.bank.get_goal(self.bank_seconds)),
            "pressure_rebasing": False,
            "repeat_after_end": False,
            "preview_after_end": "clamp_to_selected_bank_endpoint",
        }
