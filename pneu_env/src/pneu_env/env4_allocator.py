from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium.spaces import Box

from pneu_env.env4 import PneuEnv4


class PneuEnv4Allocator(PneuEnv4):
    """lib4 env with a low-dimensional task action and valve allocator.

    The SAC policy acts in a reduced action space, while this env maps that
    action to the 6 physical valve commands expected by sim4/lib4.
    """

    SUPPORTED_ACTION_MODES = {"alloc4", "alloc2"}

    def __init__(
        self,
        *args: Any,
        action_mode: str = "alloc4",
        allocator_kwargs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        if action_mode not in self.SUPPORTED_ACTION_MODES:
            raise ValueError(
                f"Unsupported allocator action_mode={action_mode!r}. "
                f"Supported modes: {sorted(self.SUPPORTED_ACTION_MODES)}"
            )

        self.action_mode = action_mode
        self.allocator_kwargs = {
            "close_raw": 0.0,
            "crack_raw": 0.45,
            "open_raw": 0.90,
            "deadband": 0.05,
            "smooth_coeff": 0.0,
        }
        if allocator_kwargs is not None:
            self.allocator_kwargs.update(allocator_kwargs)
        self._allocator_ready = False

        super().__init__(*args, **kwargs)

        if self.action_mode == "alloc2" and self.steady_chamber_ctrl is None:
            raise ValueError(
                "action_mode='alloc2' requires steady_chamber_ctrl because "
                "the policy only outputs [act_pos_effort, act_neg_effort]."
            )

        self.dim_policy_act = 4 if self.action_mode == "alloc4" else 2
        self.dim_act_traj = self.num_act * self.dim_policy_act
        self.action_space = Box(
            low=self.action_space.low.flat[0],
            high=self.action_space.high.flat[0],
            shape=(self.dim_act_traj,),
            dtype=np.float64,
        )
        self._allocator_ready = True
        self.prev_action = self._expand_ctrl_traj(self.action_space.low)[0]

    def _raw_to_signed_effort(self, raw: float) -> float:
        return float(np.clip(2.0 * float(raw) - 1.0, -1.0, 1.0))

    def _effort_to_valve_pair(self, effort: float) -> tuple[float, float]:
        cfg = self.allocator_kwargs
        close_raw = float(cfg["close_raw"])
        crack_raw = float(cfg["crack_raw"])
        open_raw = float(cfg["open_raw"])
        deadband = float(cfg["deadband"])

        effort = float(np.clip(effort, -1.0, 1.0))
        if abs(effort) < deadband:
            return close_raw, close_raw

        mag = min(max((abs(effort) - deadband) / max(1.0 - deadband, 1e-9), 0.0), 1.0)
        active_raw = crack_raw + (open_raw - crack_raw) * mag
        active_raw = float(np.clip(active_raw, 0.0, 1.0))
        close_raw = float(np.clip(close_raw, 0.0, 1.0))

        if effort > 0.0:
            return active_raw, close_raw
        return close_raw, active_raw

    def _allocate_single_action(
        self,
        task_action: np.ndarray,
        update_chamber_integral: bool,
    ) -> np.ndarray:
        task_action = np.asarray(task_action, dtype=np.float64).reshape(-1)

        if self.action_mode == "alloc4":
            if task_action.size != 4:
                raise ValueError(f"alloc4 expects 4D action, got shape {task_action.shape}")
            ch_pos_raw, ch_neg_raw, pos_raw, neg_raw = task_action
        elif self.action_mode == "alloc2":
            if task_action.size != 2:
                raise ValueError(f"alloc2 expects 2D action, got shape {task_action.shape}")
            ch_pos_raw, ch_neg_raw = self._get_chamber_ctrl(
                update_integral=update_chamber_integral
            )
            pos_raw, neg_raw = task_action
        else:
            raise RuntimeError(f"Unexpected allocator action_mode={self.action_mode!r}")

        pos_effort = self._raw_to_signed_effort(pos_raw)
        neg_effort = self._raw_to_signed_effort(neg_raw)
        pos_in_raw, pos_out_raw = self._effort_to_valve_pair(pos_effort)
        neg_in_raw, neg_out_raw = self._effort_to_valve_pair(neg_effort)

        ctrl = np.array(
            [
                ch_pos_raw,
                ch_neg_raw,
                pos_in_raw,
                pos_out_raw,
                neg_in_raw,
                neg_out_raw,
            ],
            dtype=np.float64,
        )
        return np.clip(ctrl, self.action_space.low.flat[0], self.action_space.high.flat[0])

    def _expand_ctrl_traj(
        self,
        ctrl: np.ndarray,
        update_chamber_integral: bool = False,
    ) -> np.ndarray:
        if not getattr(self, "_allocator_ready", False):
            return super()._expand_ctrl_traj(
                ctrl,
                update_chamber_integral=update_chamber_integral,
            )

        task_traj = np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_policy_act)
        ctrl_traj = [
            self._allocate_single_action(
                task_action,
                update_chamber_integral=update_chamber_integral,
            )
            for task_action in task_traj
        ]

        smooth_coeff = float(self.allocator_kwargs.get("smooth_coeff", 0.0))
        if smooth_coeff > 0.0 and len(ctrl_traj) > 0:
            smoothed = []
            prev = np.asarray(self.prev_action, dtype=np.float64).copy()
            alpha = min(max(smooth_coeff, 0.0), 1.0)
            for ctrl6 in ctrl_traj:
                ctrl6 = (1.0 - alpha) * ctrl6 + alpha * prev
                smoothed.append(ctrl6)
                prev = ctrl6
            ctrl_traj = smoothed

        return np.asarray(ctrl_traj, dtype=np.float64)
