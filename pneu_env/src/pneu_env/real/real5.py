from __future__ import annotations

from typing import Any

import numpy as np

from pneu_env.pid5 import PneumaticController5
from pneu_env.real.real4 import PneuReal as PneuReal4


class PneuReal(PneuReal4):
    """Real backend using the same five-reference controller as sim5."""

    def __init__(
        self,
        *args: Any,
        controller: dict[str, Any] | None = None,
        reserve_residual_limit: float = 20.0,
        valve_residual_limit: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.controller5 = PneumaticController5(freq=self.freq, **(controller or {}))
        self.reserve_residual_limit = float(reserve_residual_limit)
        self.valve_residual_limit = float(valve_residual_limit)
        self.last_controller_info: dict[str, Any] = {}

    def observe(self, ctrl: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict]:
        residual = np.asarray(ctrl, dtype=np.float64).reshape(-1)
        ref = np.asarray(goal, dtype=np.float64).reshape(-1)
        if residual.size not in (2, 8):
            raise ValueError(f"real5 expects 2 or 8 residual inputs, got {residual.size}")
        if ref.size < 5:
            raise ValueError(f"real5 expects 5 references, got {ref.size}")

        effective_ref = ref[:5].copy()
        effective_ref[:2] += np.clip(residual[:2], -1.0, 1.0) * self.reserve_residual_limit
        base_ctrl = self.controller5.get_action(self.obs[:4], effective_ref)
        final_ctrl = self.controller5.apply_residual(
            base_ctrl, residual, valve_residual_limit=self.valve_residual_limit
        )
        plant_goal = np.array([ref[2], ref[3], ref[4]], dtype=np.float64)
        next_obs, info = super().observe(final_ctrl, plant_goal)
        info["Observation"].update(
            ref_ch_pos=ref[0],
            ref_ch_neg=ref[1],
            effective_ref_ch_pos=effective_ref[0],
            effective_ref_ch_neg=effective_ref[1],
            base_ctrl=base_ctrl.copy(),
            residual_ctrl=residual.copy(),
            final_ctrl=final_ctrl.copy(),
        )
        self.last_controller_info = dict(self.controller5.last_info)
        info["controller5"] = self.last_controller_info
        return next_obs, info

    def reset_pid(self) -> None:
        self.controller5.reset()

    def set_controller5(self, **kwargs: Any) -> None:
        self.controller5 = PneumaticController5(freq=self.freq, **kwargs)
