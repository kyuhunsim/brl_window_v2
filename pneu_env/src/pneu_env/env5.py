from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box

from pneu_env.env4 import PneuEnv4
from pneu_ref.pneu_ref_nd import PneuRefND


class _ActuatorRefView:
    """Initialization adapter for the unchanged Env4 constructor."""

    def __init__(self, ref: Any):
        self.ref = ref

    def time_reset(self) -> None:
        if hasattr(self.ref, "time_reset"):
            self.ref.time_reset()

    def get_goal(self, curr_time: float) -> tuple[float, float, float]:
        goal = np.asarray(self.ref.get_goal(curr_time), dtype=np.float64)
        return float(goal[2]), float(goal[3]), float(goal[4])


class PneuEnv5(PneuEnv4):
    """Five-reference residual environment built on Env4's trajectory plumbing."""

    def __init__(
        self,
        obs: Any,
        ref: Any,
        pred: Any = None,
        residual_mode: str = "reserve",
        rwd_kwargs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        if residual_mode not in ("reserve", "valve"):
            raise ValueError("residual_mode must be 'reserve' or 'valve'")
        kwargs.pop("fixed_chamber_ctrl", None)
        kwargs.pop("steady_chamber_ctrl", None)
        super().__init__(
            obs=obs,
            ref=_ActuatorRefView(ref),
            pred=pred,
            rwd_kwargs=rwd_kwargs or {},
            fixed_chamber_ctrl=None,
            steady_chamber_ctrl=None,
            action_low=-1.0,
            action_high=1.0,
            **kwargs,
        )

        self.goal = PneuRefND(
            ref,
            num_prev=self.num_prev,
            num_pred=self.num_pred,
            ctrl_freq=self.obs.freq,
        )
        self.dim_ref = 5
        if self.goal.dim_ref != self.dim_ref:
            raise ValueError(f"PneuEnv5 expects 5 references, got {self.goal.dim_ref}")
        self.residual_mode = residual_mode
        self.dim_policy_act = 2 if residual_mode == "reserve" else 8
        self.dim_ref_traj = self.num_ref * self.dim_ref
        self.dim_act_traj = self.num_act * self.dim_policy_act
        self.dim_state = self.dim_obs_traj + self.dim_fut_traj + self.dim_ref_traj
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.dim_state,), dtype=np.float64
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.dim_act_traj,), dtype=np.float64
        )
        self.prev_action = np.zeros(self.dim_policy_act, dtype=np.float64)
        self._last_applied_ctrl = np.zeros(6, dtype=np.float64)

    def _expand_ctrl_traj(
        self,
        ctrl: np.ndarray,
        update_chamber_integral: bool = False,
    ) -> np.ndarray:
        del update_chamber_integral
        return np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_policy_act)

    def _extract_applied_ctrl(self, obs_info: Dict[str, Any]) -> np.ndarray:
        ctrl = super()._extract_applied_ctrl(obs_info)
        self._last_applied_ctrl = ctrl.copy()
        return ctrl

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_init = self._initial_obs_vector()
        init_obs = self.curr_full_press.copy() if self.episode_carry_state and self.has_episode_state else base_init
        if hasattr(self.obs, "set_init_press"):
            init_length, init_velocity = self._env_motion_to_sim_state(
                init_obs[self.disp_idx:self.base_obs_dim], self.obs
            )
            self.obs.set_init_press(
                init_obs[0], init_obs[1], init_obs[2], init_obs[3], init_length, init_velocity
            )
        self.goal.reset()
        self.obs_traj = np.tile(init_obs, (self.num_obs, 1))
        self.curr_full_press = init_obs.copy()
        self.t = 0.0
        self.prev_action = np.zeros(self.dim_policy_act, dtype=np.float64)
        self._last_applied_ctrl.fill(0.0)
        self.obs.reset_pid()
        if self.pred is not None:
            self.pred.reset_pid()
        state, _, _, _, info = self.step(np.zeros(self.dim_act_traj, dtype=np.float64))
        self.t = info["obs"]["curr_time"]
        self.obs.reset_pid()
        if self.pred is not None:
            self.pred.reset_pid()
        self.prev_action.fill(0.0)
        return state, info

    def close(self) -> None:
        residual = np.zeros(self.dim_policy_act, dtype=np.float64)
        goal = np.array([150.0, 50.0, 120.0, 80.0, 0.0], dtype=np.float64)
        self.obs.observe(residual, goal)

    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, dict[str, float]]:
        obs_traj, pred_traj, ref_traj = self._split_state(state)
        obs = obs_traj[-1]
        ref = ref_traj[self.num_obs - 1]

        ch_pos_short = max(ref[0] - obs[0], 0.0)
        ch_neg_short = max(obs[1] - ref[1], 0.0)
        act_error = ref[2:4] - obs[2:4]
        disp_error = ref[4] - obs[self.disp_idx]
        diff_error = (ref[2] - ref[3]) - (obs[2] - obs[3])

        cfg = self.rwd_kwargs
        terms = dict(
            ch_pos_shortage_reward=-float(cfg.get("ch_pos_shortage_rwd_coeff", 0.0)) * ch_pos_short**2,
            ch_neg_shortage_reward=-float(cfg.get("ch_neg_shortage_rwd_coeff", 0.0)) * ch_neg_short**2,
            pos_curr_reward=-float(cfg.get("pos_curr_rwd_coeff", 0.0)) * abs(act_error[0]),
            neg_curr_reward=-float(cfg.get("neg_curr_rwd_coeff", 0.0)) * abs(act_error[1]),
            disp_curr_reward=-float(cfg.get("disp_curr_rwd_coeff", 0.0)) * abs(disp_error),
            diff_curr_reward=-float(cfg.get("diff_curr_rwd_coeff", 0.0)) * abs(diff_error),
        )

        applied_opening = self._last_applied_ctrl.copy()
        if getattr(self.obs, "scale", False):
            span = max(float(self.obs.scale_high-self.obs.scale_low), 1e-9)
            applied_opening = np.clip((applied_opening-self.obs.scale_low) / span, 0.0, 1.0)
        conflict = float(
            applied_opening[2] * applied_opening[3]
            + applied_opening[4] * applied_opening[5]
        )
        residual = np.asarray(action, dtype=np.float64).reshape(-1, self.dim_policy_act)[0]
        action_delta = float(np.mean((residual - self.prev_action) ** 2))
        residual_magnitude = float(np.mean(residual**2))
        terms.update(
            conflict_reward=-float(cfg.get("conflict_rwd_coeff", 0.0)) * conflict,
            action_delta_reward=-float(cfg.get("action_delta_rwd_coeff", 0.0)) * action_delta,
            residual_reward=-float(cfg.get("residual_rwd_coeff", 0.0)) * residual_magnitude,
        )

        # Future terms use the predictor only when it is enabled.
        pred_tracking_reward = 0.0
        if pred_traj is not None and len(pred_traj):
            pred_ref = ref_traj[self.num_obs:self.num_obs + self.num_act][-1]
            pred_obs = pred_traj[-1]
            pred_tracking_reward = -float(cfg.get("pred_rwd_coeff", 0.0)) * (
                abs(pred_ref[2] - pred_obs[2])
                + abs(pred_ref[3] - pred_obs[3])
                + abs(pred_ref[4] - pred_obs[self.disp_idx])
            )
        terms["pred_tracking_reward"] = float(pred_tracking_reward)

        reward = float(sum(terms.values()))
        info = dict(
            **terms,
            ch_pos_shortage=float(ch_pos_short),
            ch_neg_shortage=float(ch_neg_short),
            pos_error=float(act_error[0]),
            neg_error=float(act_error[1]),
            disp_error=float(disp_error),
            diff_error=float(diff_error),
            conflict=conflict,
            action_delta=action_delta,
            residual_magnitude=residual_magnitude,
        )
        return reward, info

    def set_controller5(self, **kwargs: Any) -> None:
        self.obs.set_controller5(**kwargs)
        if self.pred is not None:
            self.pred.set_controller5(**kwargs)

    def verbose(self, info: Dict[str, Any]) -> None:
        obs = info["obs"]
        reward = info["reward"]
        print(
            "[ INFO] Pneumatic Env5 ==> "
            f"t={obs['curr_time']:.3f} "
            f"ch=({obs['sen_pos']:.2f},{obs['sen_neg']:.2f}) "
            f"act=({obs['sen_act_pos']:.2f},{obs['sen_act_neg']:.2f}) "
            f"disp={obs.get('sen_env_displacement', 0.0):.2f} "
            f"reward={sum(v for k, v in reward.items() if k.endswith('_reward')):.3f}"
        )
