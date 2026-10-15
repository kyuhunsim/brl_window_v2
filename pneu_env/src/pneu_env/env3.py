import sys

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from gymnasium.spaces import Box

from pneu_ref.pneu_ref import PneuRef


class PneuEnv3:
    def __init__(
        self,
        obs: Any,
        ref: Any,
        pred: Any = None,
        num_prev: int = 10,
        num_pred: int = 10,
        num_act: int = 1,
        rwd_kwargs: Dict[str, Any] = dict(
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_curr_rwd_coeff=0.0,
            neg_curr_rwd_coeff=0.0,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
            action_delta_rwd_coeff=0.0,
            second_action_delta_rwd_coeff=0.0,
            conflict_rwd_coeff=0.0,
            chamber_reserve_rwd_coeff=0.0,
            chamber_margin_kpa=15.0,
            chamber_deadband_kpa=5.0,
            chamber_bound_rwd_coeff=0.0,
            chamber_pos_min_kpa=101.325,
            chamber_pos_max_kpa=220.0,
            chamber_neg_min_kpa=10.0,
            chamber_neg_max_kpa=101.325,
            # Optional nonlinear/delay-aware shaping.  Defaults preserve the
            # historical pressure-only absolute-error reward.
            tracking_loss_mode="abs",
            tracking_huber_delta=1.0,
            pos_rate_rwd_coeff=0.0,
            neg_rate_rwd_coeff=0.0,
            pos_rate_scale_kpa_s=12.0,
            neg_rate_scale_kpa_s=6.0,
            tracking_channels="both",
        ),
        pos_pred_rnd_offset_range: float = 0,
        neg_pred_rnd_offset_range: float = 0,
        verbose: bool = True,
        action_low: float = 0.0,
        action_high: float = 1.0,
        episode_carry_state: bool = True,
        episode_carry_blocks: int = 0,
        action_history_len: int = 0,
        fixed_chamber_ctrl: Optional[Tuple[float, float]] = None,
        steady_chamber_ctrl: Optional[Dict[str, float]] = None,
        dynamic_chamber_ctrl: Optional[Dict[str, float]] = None,
        fixed_action_mask: Optional[Tuple[bool, ...]] = None,
        fixed_action_values: Optional[Tuple[float, ...]] = None,
        include_valve_z: bool = False,
        include_physics_features: bool = False,
        physics_feature_mode: str = "all",
        physics_rate_scale_kpa_s: float = 50.0,
        include_valve_diagnostics: bool = False,
        include_integral_error: bool = False,
        integral_error_decay: float = 1.0,
        integral_error_limit: float = 100.0,
        integral_error_scale: float = 20.0,
        action_chunk_steps: int = 1,
        action_chunk_gamma: float = 0.99,
    ):
        self.obs = obs
        self.goal = PneuRef(
            ref,
            num_prev=num_prev,
            num_pred=num_pred,
            ctrl_freq=self.obs.freq,
        )
        self.pred = pred

        self.num_prev = num_prev
        self.num_pred = num_pred
        self.num_act = num_act
        self.action_chunk_steps = max(int(action_chunk_steps), 1)
        if self.action_chunk_steps > 1 and self.num_act > 1:
            raise ValueError(
                "action_chunk_steps>1 cannot be combined with num_act>1"
            )
        self.action_chunk_gamma = float(action_chunk_gamma)
        if not 0.0 < self.action_chunk_gamma <= 1.0:
            raise ValueError("action_chunk_gamma must lie in (0, 1]")

        self.num_obs = num_prev + 1
        self.num_ref = num_prev + num_pred + 1

        self.dim_obs = 4
        self.dim_ref = 2
        self.dim_act = 6
        self.action_history_len = max(int(action_history_len), 0)
        self.dim_action_history = self.action_history_len * self.dim_act
        # The simulator exposes one hysteresis state z per valve.  It is
        # diagnostic-only by default; the privileged ablation can append a
        # signed-log-compressed version to the policy observation.
        self.include_valve_z = bool(include_valve_z)
        self.dim_valve_z = self.dim_act if self.include_valve_z else 0
        self.include_physics_features = bool(include_physics_features)
        self.physics_feature_mode = str(physics_feature_mode).lower()
        physics_feature_dims = {
            "none": 0,
            "rate": 4,
            "zero4": 4,
            "reserve": 2,
            "rate_reserve": 6,
            # Explicit spelling of history-rate-reserve.
            "hrr": 6,
            # Tracking-error derivative and its HRR combination.
            "edot": 2,
            "rate_reserve_edot": 8,
            # Fixed P/I/D tracking-error summary.
            "pide": 6,
            # GPIDE-style generalized difference/action summary for lib3:
            # 4 observation differences + 6 current actions + 6 action deltas.
            "gpide": 16,
            "all": 18,
        }
        if self.physics_feature_mode not in physics_feature_dims:
            raise ValueError(
                "physics_feature_mode must be one of "
                "none, rate, zero4, reserve, rate_reserve, hrr, edot, "
                "rate_reserve_edot, pide, gpide, all"
            )
        self.physics_rate_scale_kpa_s = float(physics_rate_scale_kpa_s)
        if (
            not np.isfinite(self.physics_rate_scale_kpa_s)
            or self.physics_rate_scale_kpa_s <= 0.0
        ):
            raise ValueError("physics_rate_scale_kpa_s must be finite and positive")
        self.dim_physics_features = (
            physics_feature_dims[self.physics_feature_mode]
            if self.include_physics_features
            else 0
        )
        self.include_valve_diagnostics = bool(include_valve_diagnostics)
        self.dim_valve_diagnostics = 18 if self.include_valve_diagnostics else 0
        self.include_integral_error = bool(include_integral_error)
        self.dim_integral_error = 2 if self.include_integral_error else 0
        self.integral_error_decay = float(integral_error_decay)
        if not 0.0 < self.integral_error_decay <= 1.0:
            raise ValueError("integral_error_decay must lie in (0, 1]")
        self.integral_error_limit = max(float(integral_error_limit), 1e-9)
        self.integral_error_scale = max(float(integral_error_scale), 1e-9)
        self.fixed_chamber_ctrl = None
        self.steady_chamber_ctrl = None
        self.fixed_action_mask = None
        self.fixed_action_values = None
        if fixed_action_mask is not None and (
            fixed_chamber_ctrl is not None
            or steady_chamber_ctrl is not None
            or dynamic_chamber_ctrl is not None
        ):
            raise ValueError(
                "fixed_action_mask cannot be combined with fixed_chamber_ctrl "
                "or steady_chamber_ctrl"
            )
        if fixed_action_mask is not None:
            mask = np.asarray(fixed_action_mask, dtype=bool)
            if mask.shape != (self.dim_act,):
                raise ValueError(
                    "fixed_action_mask must have shape (6,), "
                    f"got {mask.shape}"
                )
            if not np.any(mask):
                raise ValueError("fixed_action_mask must fix at least one control")
            values = (
                np.zeros(self.dim_act, dtype=np.float64)
                if fixed_action_values is None
                else np.asarray(fixed_action_values, dtype=np.float64)
            )
            if values.shape != (self.dim_act,):
                raise ValueError(
                    "fixed_action_values must have shape (6,), "
                    f"got {values.shape}"
                )
            if np.any(values[mask] < action_low) or np.any(values[mask] > action_high):
                raise ValueError(
                    "fixed_action_values for fixed controls must lie within "
                    f"[{action_low}, {action_high}], got {values[mask]}"
                )
            self.fixed_action_mask = mask
            self.fixed_action_values = values
        chamber_controller_count = sum(
            value is not None
            for value in (fixed_chamber_ctrl, steady_chamber_ctrl, dynamic_chamber_ctrl)
        )
        if chamber_controller_count > 1:
            raise ValueError(
                "fixed_chamber_ctrl, steady_chamber_ctrl, and dynamic_chamber_ctrl "
                "are mutually exclusive"
            )
        if fixed_chamber_ctrl is not None:
            fixed_chamber_ctrl_arr = np.asarray(fixed_chamber_ctrl, dtype=np.float64)
            if fixed_chamber_ctrl_arr.shape != (2,):
                raise ValueError(
                    "fixed_chamber_ctrl must contain two normalized controls "
                    f"for the positive/negative chamber valves, got shape {fixed_chamber_ctrl_arr.shape}"
                )
            if np.any(fixed_chamber_ctrl_arr < action_low) or np.any(fixed_chamber_ctrl_arr > action_high):
                raise ValueError(
                    f"fixed_chamber_ctrl must be within [{action_low}, {action_high}], "
                    f"got {fixed_chamber_ctrl_arr}"
                )
            self.fixed_chamber_ctrl = fixed_chamber_ctrl_arr
        if steady_chamber_ctrl is not None:
            required_keys = {
                "pos_target",
                "neg_target",
                "kp",
                "ki",
                "deadband",
                "integral_limit",
                "min_open",
                "max_open",
            }
            missing_keys = required_keys - set(steady_chamber_ctrl)
            if missing_keys:
                raise ValueError(f"steady_chamber_ctrl is missing keys: {sorted(missing_keys)}")
            self.steady_chamber_ctrl = {
                key: float(steady_chamber_ctrl[key])
                for key in required_keys
            }
        self.dynamic_chamber_ctrl = None
        if dynamic_chamber_ctrl is not None:
            required_keys = {
                "pos_margin",
                "neg_margin",
                "pos_min",
                "pos_max",
                "neg_min",
                "neg_max",
                "kp",
                "ki",
                "deadband",
                "integral_limit",
                "min_open",
                "max_open",
            }
            missing_keys = required_keys - set(dynamic_chamber_ctrl)
            if missing_keys:
                raise ValueError(
                    "dynamic_chamber_ctrl is missing keys: "
                    f"{sorted(missing_keys)}"
                )
            self.dynamic_chamber_ctrl = {
                key: float(dynamic_chamber_ctrl[key])
                for key in required_keys
            }
        if self.fixed_action_mask is not None:
            self.dim_policy_act = int(np.count_nonzero(~self.fixed_action_mask))
        else:
            self.dim_policy_act = (
                4
                if (
                    self.fixed_chamber_ctrl is not None
                    or self.steady_chamber_ctrl is not None
                    or self.dynamic_chamber_ctrl is not None
                )
                else self.dim_act
            )

        self.dim_obs_traj = self.num_obs * self.dim_obs
        self.dim_fut_traj = self.num_act * self.dim_obs if pred is not None else 0
        self.dim_ref_traj = self.num_ref * self.dim_ref
        self.action_output_steps = (
            self.action_chunk_steps if self.action_chunk_steps > 1 else self.num_act
        )
        self.dim_act_traj = self.action_output_steps * self.dim_policy_act
        self.dim_state = (
            self.dim_obs_traj
            + self.dim_action_history
            + self.dim_valve_z
            + self.dim_physics_features
            + self.dim_valve_diagnostics
            + self.dim_integral_error
            + self.dim_fut_traj
            + self.dim_ref_traj
        )

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dim_state,),
            dtype=np.float64,
        )
        self.action_space = Box(
            low=action_low,
            high=action_high,
            shape=(self.dim_act_traj,),
            dtype=np.float64,
        )

        init_press = np.array(
            [
                getattr(self.obs, "init_pos_press", 101.325),
                getattr(self.obs, "init_neg_press", 101.325),
                getattr(self.obs, "init_act_pos_press", 101.325),
                getattr(self.obs, "init_act_neg_press", 101.325),
            ],
            dtype=np.float64,
        )
        self.obs_traj = np.tile(init_press, (self.num_obs, 1))
        self.curr_full_press = init_press.copy()
        self.prev_full_press = init_press.copy()
        self.prev_ref_current = np.array([101.325, 101.325], dtype=np.float64)
        self.t = 0.0
        self.rwd_kwargs = rwd_kwargs
        self.offset_range = np.array(
            [pos_pred_rnd_offset_range, neg_pred_rnd_offset_range],
            dtype=np.float64,
        )
        self.publish_obs = False
        self.is_pid = False
        self.verbose_enabled = bool(verbose)
        self.episode_carry_state = bool(episode_carry_state)
        self.episode_carry_blocks = max(int(episode_carry_blocks), 0)
        self.carry_block_index = 0
        self.hard_reset_count = 0
        self.force_hard_reset_next = False
        self.has_episode_state = False
        self.chamber_integral_err = np.zeros(2, dtype=np.float64)
        self.actuator_integral_error = np.zeros(2, dtype=np.float64)
        self.prev_action = self._expand_ctrl_traj(self.action_space.low)[0]
        self.prev_prev_action = self.prev_action.copy()
        self.action_history = np.tile(
            self.prev_action,
            (self.action_history_len, 1),
        )
        self.last_policy_action = np.asarray(self.action_space.low, dtype=np.float64).copy()
        self.current_chamber_target = np.array([np.nan, np.nan], dtype=np.float64)

        # Fixed feature scales are used only when SAC explicitly enables
        # observation normalization.  Reward calculations continue to use the
        # original physical units held in ``state``.
        physics_scale = np.empty(0, dtype=np.float64)
        if self.include_physics_features:
            rate_scale = np.full(
                4,
                self.physics_rate_scale_kpa_s,
                dtype=np.float64,
            )
            reserve_scale = np.full(2, 100.0, dtype=np.float64)
            action_scale = np.ones(12, dtype=np.float64)
            if self.physics_feature_mode in ("rate", "zero4"):
                physics_scale = rate_scale
            elif self.physics_feature_mode == "reserve":
                physics_scale = reserve_scale
            elif self.physics_feature_mode == "rate_reserve":
                physics_scale = np.r_[rate_scale, reserve_scale]
            elif self.physics_feature_mode == "hrr":
                physics_scale = np.r_[rate_scale, reserve_scale]
            elif self.physics_feature_mode == "edot":
                physics_scale = np.full(2, self.physics_rate_scale_kpa_s, dtype=np.float64)
            elif self.physics_feature_mode == "rate_reserve_edot":
                physics_scale = np.r_[
                    rate_scale,
                    reserve_scale,
                    np.full(2, self.physics_rate_scale_kpa_s, dtype=np.float64),
                ]
            elif self.physics_feature_mode == "pide":
                # error [kPa], integral error [kPa*s], error derivative [kPa/s]
                physics_scale = np.r_[
                    np.full(2, 100.0, dtype=np.float64),
                    np.full(2, self.integral_error_scale, dtype=np.float64),
                    np.full(2, self.physics_rate_scale_kpa_s, dtype=np.float64),
                ]
            elif self.physics_feature_mode == "gpide":
                physics_scale = np.r_[
                    np.full(4, 100.0, dtype=np.float64),
                    np.ones(12, dtype=np.float64),
                ]
            elif self.physics_feature_mode == "all":
                physics_scale = np.r_[rate_scale, reserve_scale, action_scale]
        self.policy_state_scale = np.r_[
            np.full(self.dim_obs_traj, 100.0, dtype=np.float64),
            np.ones(self.dim_action_history, dtype=np.float64),
            np.ones(self.dim_valve_z, dtype=np.float64),
            physics_scale,
            np.ones(self.dim_valve_diagnostics, dtype=np.float64),
            np.ones(self.dim_integral_error, dtype=np.float64),
            np.full(self.dim_fut_traj, 100.0, dtype=np.float64),
            np.full(self.dim_ref_traj, 100.0, dtype=np.float64),
        ]
        if self.policy_state_scale.shape != (self.dim_state,):
            raise RuntimeError(
                "lib3 policy-state scale does not match observation dimension: "
                f"{self.policy_state_scale.shape} != {(self.dim_state,)}"
            )

    def _expand_ctrl_traj(
        self,
        ctrl: np.ndarray,
        update_chamber_integral: bool = False,
    ) -> np.ndarray:
        ctrl_traj = np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_policy_act)
        if self.fixed_action_mask is not None:
            full_ctrl = np.tile(self.fixed_action_values, (len(ctrl_traj), 1))
            full_ctrl[:, ~self.fixed_action_mask] = ctrl_traj
            return full_ctrl
        if (
            self.fixed_chamber_ctrl is None
            and self.steady_chamber_ctrl is None
            and self.dynamic_chamber_ctrl is None
        ):
            return ctrl_traj
        chamber_ctrl = np.tile(
            self._get_chamber_ctrl(update_integral=update_chamber_integral),
            (len(ctrl_traj), 1),
        )
        return np.c_[chamber_ctrl, ctrl_traj]

    def _compute_steady_chamber_ctrl(
        self,
        chamber_press: np.ndarray,
        integral_err: np.ndarray,
        integrate: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.steady_chamber_ctrl
        if cfg is None:
            raise RuntimeError("steady chamber controller is not configured")

        ch_pos, ch_neg = np.asarray(chamber_press, dtype=np.float64)
        signed_err = np.array(
            [
                ch_pos - cfg["pos_target"],
                cfg["neg_target"] - ch_neg,
            ],
            dtype=np.float64,
        )
        if integrate:
            integral_err = np.clip(
                integral_err + signed_err / self.obs.freq,
                0.0,
                cfg["integral_limit"],
            )
        proportional_err = np.where(signed_err > cfg["deadband"], signed_err, 0.0)
        effort = cfg["kp"] * proportional_err + cfg["ki"] * integral_err
        ctrl = np.where(
            effort > 0.0,
            np.clip(effort, cfg["min_open"], cfg["max_open"]),
            0.0,
        )
        return ctrl, integral_err

    def _compute_dynamic_chamber_ctrl(
        self,
        chamber_press: np.ndarray,
        actuator_ref: np.ndarray,
        integral_err: np.ndarray,
        integrate: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Control chamber valves against the current actuator reference.

        The controller maintains a reserve above the positive actuator target
        and below the negative actuator target.  The valve directions match
        the calibrated lib3 model: positive chamber command vents excess
        pressure, while negative chamber command raises insufficient pressure.
        The policy still controls only the four actuator-side valves.
        """
        cfg = self.dynamic_chamber_ctrl
        if cfg is None:
            raise RuntimeError("dynamic chamber controller is not configured")
        actuator_ref = np.asarray(actuator_ref, dtype=np.float64).reshape(2)
        target = np.array(
            [
                np.clip(
                    actuator_ref[0] + cfg["pos_margin"],
                    cfg["pos_min"],
                    cfg["pos_max"],
                ),
                np.clip(
                    actuator_ref[1] - cfg["neg_margin"],
                    cfg["neg_min"],
                    cfg["neg_max"],
                ),
            ],
            dtype=np.float64,
        )
        ch_pos, ch_neg = np.asarray(chamber_press, dtype=np.float64)
        signed_err = np.array(
            [ch_pos - target[0], target[1] - ch_neg],
            dtype=np.float64,
        )
        if integrate:
            integral_err = np.clip(
                integral_err + signed_err / self.obs.freq,
                0.0,
                cfg["integral_limit"],
            )
        proportional_err = np.where(signed_err > cfg["deadband"], signed_err, 0.0)
        effort = cfg["kp"] * proportional_err + cfg["ki"] * integral_err
        ctrl = np.where(
            effort > 0.0,
            np.clip(effort, cfg["min_open"], cfg["max_open"]),
            0.0,
        )
        return ctrl, integral_err, target

    def _current_actuator_reference(self) -> np.ndarray:
        """Return the current actuator target already held by PneuRef."""
        return self.goal.buf.reshape(-1, self.dim_ref)[self.num_obs - 1].copy()

    def _get_chamber_ctrl(
        self,
        chamber_press: Optional[np.ndarray] = None,
        update_integral: bool = False,
    ) -> np.ndarray:
        if self.fixed_chamber_ctrl is not None:
            self.current_chamber_target = np.array([np.nan, np.nan], dtype=np.float64)
            return self.fixed_chamber_ctrl.copy()
        if self.steady_chamber_ctrl is not None:
            if chamber_press is None:
                chamber_press = self.curr_full_press[0:2]
            ctrl, integral_err = self._compute_steady_chamber_ctrl(
                chamber_press,
                self.chamber_integral_err,
                integrate=update_integral,
            )
            self.current_chamber_target = np.array(
                [
                    self.steady_chamber_ctrl["pos_target"],
                    self.steady_chamber_ctrl["neg_target"],
                ],
                dtype=np.float64,
            )
            if update_integral:
                self.chamber_integral_err = integral_err
            return ctrl
        if self.dynamic_chamber_ctrl is None:
            raise RuntimeError("chamber controller is not configured")

        if chamber_press is None:
            chamber_press = self.curr_full_press[0:2]
        ctrl, integral_err, target = self._compute_dynamic_chamber_ctrl(
            chamber_press,
            self._current_actuator_reference(),
            self.chamber_integral_err,
            integrate=update_integral,
        )
        self.current_chamber_target = target
        if update_integral:
            self.chamber_integral_err = integral_err
        return ctrl

    def _append_action_history(self, applied_action: np.ndarray) -> np.ndarray:
        """Return action history ending in the command applied this step."""
        if self.action_history_len == 0:
            return np.empty((0, self.dim_act), dtype=np.float64)
        applied = np.asarray(applied_action, dtype=np.float64).reshape(1, self.dim_act)
        if self.action_history_len == 1:
            return applied.copy()
        return np.r_[self.action_history[1:], applied]

    def _get_integral_error_observation(self) -> np.ndarray:
        if not self.include_integral_error:
            return np.empty(0, dtype=np.float64)
        # The integral is kept in kPa*s and exposed through a bounded,
        # dimensionless feature so it does not dominate pressure history.
        return np.clip(
            self.actuator_integral_error / self.integral_error_scale,
            -5.0,
            5.0,
        )

    def request_hard_reset(self) -> None:
        """Make the next episode boundary reset the complete lib3 plant."""
        self.force_hard_reset_next = True

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        # A carried episode is a continuation of the physical rollout.  Keep
        # the lib3 simulator, valve states, pump states, reference clock, and
        # observation history untouched; only the RL horizon boundary ends.
        can_carry = (
            self.episode_carry_state
            and self.has_episode_state
            and not self.force_hard_reset_next
            and (
                self.episode_carry_blocks == 0
                or self.carry_block_index < self.episode_carry_blocks
            )
        )
        if can_carry:
            self.carry_block_index += 1
            state, state_info = self._state_from_current(self.last_policy_action)
            return state, self.get_info(state_info, {})

        base_init_press = np.array(
            [
                getattr(self.obs, "init_pos_press", 101.325),
                getattr(self.obs, "init_neg_press", 101.325),
                getattr(self.obs, "init_act_pos_press", 101.325),
                getattr(self.obs, "init_act_neg_press", 101.325),
            ],
            dtype=np.float64,
        )
        init_press = base_init_press
        self.force_hard_reset_next = False
        if hasattr(self.obs, "set_init_press"):
            self.obs.set_init_press(
                init_press[0],
                init_press[1],
                init_press[2],
                init_press[3],
            )
        if hasattr(self.goal, "reset"):
            self.goal.reset(initial_state=init_press[:2])
        self.obs_traj = np.tile(init_press, (self.num_obs, 1))
        self.curr_full_press = init_press.copy()
        self.prev_full_press = init_press.copy()
        self.prev_ref_current = np.array([101.325, 101.325], dtype=np.float64)
        self.t = 0.0
        self.chamber_integral_err = np.zeros(2, dtype=np.float64)
        self.actuator_integral_error = np.zeros(2, dtype=np.float64)
        self.current_chamber_target = np.array([np.nan, np.nan], dtype=np.float64)
        self.prev_action = self._expand_ctrl_traj(self.action_space.low)[0]
        self.prev_prev_action = self.prev_action.copy()
        self.action_history = np.tile(
            self.prev_action,
            (self.action_history_len, 1),
        )
        self.last_policy_action = np.asarray(self.action_space.low, dtype=np.float64).copy()
        self.has_episode_state = False
        self.carry_block_index = 1
        self.hard_reset_count += 1

        if self.is_pid:
            self.obs.reset_pid()

        # Establish the initial policy state without injecting
        # action_space.low into the pneumatic simulator before the first RL
        # action is selected.
        state, state_info = self._state_from_current(self.last_policy_action)
        return state, self.get_info(state_info, {})

    def _step_single(
        self,
        action: np.ndarray,
        emit_verbose: bool = True,
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state, state_info = self.get_state(action)
        reward, reward_info = self.get_reward(
            state=state,
            action=action,
        )
        terminated = False
        truncated = False

        info = self.get_info(state_info, reward_info)
        if self.verbose_enabled and emit_verbose:
            self.verbose(info)
        if self.publish_obs:
            self.publish_observation(info)

        applied_action = np.asarray(state_info["ctrl"], dtype=np.float64)[0].copy()
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = applied_action

        return state, reward, terminated, truncated, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.action_chunk_steps <= 1:
            return self._step_single(action)

        raw_action = np.asarray(action, dtype=np.float64).reshape(
            -1, self.dim_policy_act
        )
        if len(raw_action) != self.action_chunk_steps:
            raise ValueError(
                "chunked action must contain exactly "
                f"{self.action_chunk_steps} policy commands, got {raw_action.shape}"
            )

        total_reward = 0.0
        substep_rewards: list[float] = []
        substep_infos: list[dict[str, Any]] = []
        final_state = None
        final_terminated = False
        final_truncated = False
        final_info = None
        for chunk_index, sub_action in enumerate(raw_action):
            state, reward, terminated, truncated, info = self._step_single(
                sub_action,
                emit_verbose=(chunk_index == self.action_chunk_steps - 1),
            )
            total_reward += (self.action_chunk_gamma ** chunk_index) * float(reward)
            substep_rewards.append(float(reward))
            # Keep the physical substep records for evaluators.  The returned
            # policy state remains the final substep state, so this does not
            # change the transition seen by SAC; it only prevents a K>1
            # validation trace from silently being down-sampled.
            substep_infos.append(info)
            final_state = state
            final_terminated = terminated
            final_truncated = truncated
            final_info = info

        assert final_state is not None and final_info is not None
        final_info["chunk_steps"] = int(self.action_chunk_steps)
        final_info["chunk_substep_rewards"] = np.asarray(
            substep_rewards, dtype=np.float64
        )
        final_info["chunk_substep_infos"] = substep_infos
        final_info["chunk_reward"] = float(total_reward)
        final_info["reward"]["chunk_steps"] = int(self.action_chunk_steps)
        final_info["reward"]["chunk_reward"] = float(total_reward)
        return (
            final_state,
            float(total_reward),
            final_terminated,
            final_truncated,
            final_info,
        )

    def close(self) -> None:
        ctrl = np.zeros(self.dim_act, dtype=np.float64)
        goal = np.array([101.325, 101.325], dtype=np.float64)
        _, _ = self.obs.observe(ctrl, goal)

    def get_state(
        self,
        ctrl: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl_traj = self._expand_ctrl_traj(ctrl, update_chamber_integral=True)
        previous_full_press = self.curr_full_press.copy()
        previous_ref_current = self.goal.buf.reshape(-1, self.dim_ref)[self.num_obs - 1].copy()
        goal_traj = self.goal.get_ref(self.t).reshape(-1, self.dim_ref)

        obs, obs_info = self.obs.observe(
            ctrl_traj.copy()[0],
            goal_traj.copy()[self.num_obs - 1],
        )
        full_press = obs[1:5].reshape(1, self.dim_obs)
        obs_traj = np.r_[
            self.obs_traj[1:],
            full_press,
        ]

        pred_info = None
        valve_z = self._get_valve_z_observation()
        valve_diagnostics = self._get_valve_diagnostics_observation()
        next_action_history = self._append_action_history(ctrl_traj[0])
        physics_features = self._get_physics_features(
            obs_traj,
            goal_traj,
            action_history=(ctrl_traj[0], self.prev_action),
        )
        self.actuator_integral_error = np.clip(
            self.integral_error_decay * self.actuator_integral_error
            + (
                goal_traj[self.num_obs - 1] - full_press[0, 2:4]
            ) / max(float(self.obs.freq), 1e-9),
            -self.integral_error_limit,
            self.integral_error_limit,
        )
        integral_error = self._get_integral_error_observation()
        state_prefix = np.r_[
            obs_traj.reshape(-1),
            next_action_history.reshape(-1),
            valve_z,
            physics_features,
            valve_diagnostics,
            integral_error,
        ]
        if self.pred is not None:
            pred_traj, pred_info = self.predict_obs(
                init_press=obs_traj[-1],
                ctrl_traj=ctrl_traj,
                ref_traj=goal_traj[self.num_obs:self.num_obs + self.num_act],
            )
            state = np.r_[
                state_prefix,
                pred_traj.reshape(-1),
                goal_traj.reshape(-1),
            ]
        else:
            state = np.r_[state_prefix, goal_traj.reshape(-1)]

        self.t = obs[0]
        self.prev_full_press = previous_full_press
        self.prev_ref_current = previous_ref_current
        self.obs_traj = obs_traj
        self.curr_full_press = obs_traj[-1].copy()
        self.action_history = next_action_history
        self.has_episode_state = True
        self.last_policy_action = np.asarray(ctrl, dtype=np.float64).copy()
        obs_info["ctrl"] = ctrl_traj
        obs_info["chamber_ctrl"] = ctrl_traj[0, :2].copy()
        obs_info["chamber_target"] = self.current_chamber_target.copy()
        obs_info["pred"] = pred_info
        obs_info["goal_traj"] = goal_traj
        obs_info["full_press"] = self.curr_full_press.copy()
        # z/force/area/mdot are simulator diagnostics, not policy inputs.
        # Keeping them in info lets us verify whether an oscillation is caused
        # by the hidden valve state without giving the policy privileged data.
        if hasattr(self.obs, "get_valve_debug"):
            obs_info["valve_debug"] = self.obs.get_valve_debug()
        if self.include_valve_z:
            obs_info["valve_z_observation"] = valve_z.copy()
        if self.include_physics_features:
            obs_info["physics_features"] = physics_features.copy()
        if self.include_valve_diagnostics:
            obs_info["valve_diagnostics_observation"] = valve_diagnostics.copy()
        if self.include_integral_error:
            obs_info["integral_error_observation"] = integral_error.copy()

        return state, obs_info

    def _state_from_current(
        self,
        ctrl: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build a policy state without advancing the lib3 simulator."""
        ctrl_traj = self._expand_ctrl_traj(
            np.asarray(ctrl, dtype=np.float64),
            update_chamber_integral=False,
        )
        goal_traj = np.asarray(self.goal.buf, dtype=np.float64).copy().reshape(
            -1, self.dim_ref
        )
        obs_traj = np.asarray(self.obs_traj, dtype=np.float64).copy()
        obs_traj[-1] = np.asarray(self.curr_full_press, dtype=np.float64)

        pred_info = None
        valve_z = self._get_valve_z_observation()
        valve_diagnostics = self._get_valve_diagnostics_observation()
        physics_features = self._get_physics_features(
            obs_traj,
            goal_traj,
            action_history=(self.prev_action, self.prev_prev_action),
        )
        integral_error = self._get_integral_error_observation()
        state_prefix = np.r_[
            obs_traj.reshape(-1),
            self.action_history.reshape(-1),
            valve_z,
            physics_features,
            valve_diagnostics,
            integral_error,
        ]
        if self.pred is not None:
            pred_traj, pred_info = self.predict_obs(
                init_press=obs_traj[-1],
                ctrl_traj=ctrl_traj,
                ref_traj=goal_traj[self.num_obs:self.num_obs + self.num_act],
            )
            state = np.r_[
                state_prefix,
                pred_traj.reshape(-1),
                goal_traj.reshape(-1),
            ]
        else:
            state = np.r_[state_prefix, goal_traj.reshape(-1)]

        self.obs_traj = obs_traj
        state_info = dict(
            Observation=dict(
                curr_time=float(self.t),
                sen_pos=float(obs_traj[-1, 0]),
                sen_neg=float(obs_traj[-1, 1]),
                sen_act_pos=float(obs_traj[-1, 2]),
                sen_act_neg=float(obs_traj[-1, 3]),
                ref_pos=101.325,
                ref_neg=101.325,
                ref_act_pos=float(goal_traj[self.num_obs - 1, 0]),
                ref_act_neg=float(goal_traj[self.num_obs - 1, 1]),
                ctrl_pos=float(ctrl_traj[0, 0]),
                ctrl_neg=float(ctrl_traj[0, 1]),
                ctrl_act_pos_in=float(ctrl_traj[0, 2]),
                ctrl_act_pos_out=float(ctrl_traj[0, 3]),
                ctrl_act_neg_in=float(ctrl_traj[0, 4]),
                ctrl_act_neg_out=float(ctrl_traj[0, 5]),
            ),
            **{
                "obs_w/o_noise": np.r_[self.t, obs_traj[-1]].copy(),
                "ctrl": ctrl_traj.copy(),
                "chamber_ctrl": ctrl_traj[0, :2].copy(),
                "chamber_target": self.current_chamber_target.copy(),
                "pred": pred_info,
                "goal_traj": goal_traj,
                "full_press": obs_traj[-1].copy(),
            },
        )
        return state, state_info

    def _get_valve_z_observation(self) -> np.ndarray:
        """Return a bounded privileged observation of the six valve z states.

        The raw z states span several orders of magnitude across the six
        calibrated valve models.  A signed-log transform preserves direction
        and hysteresis magnitude while keeping the policy input numerically
        well-scaled.  The transform is only enabled for the explicit
        privileged-observation ablation.
        """
        if not self.include_valve_z:
            return np.empty(0, dtype=np.float64)
        if not hasattr(self.obs, "get_valve_debug"):
            raise RuntimeError("include_valve_z requires simulator valve diagnostics")
        debug = self.obs.get_valve_debug()
        names = (
            "ch_pos_z",
            "ch_neg_z",
            "act_pos_in_z",
            "act_pos_out_z",
            "act_neg_in_z",
            "act_neg_out_z",
        )
        raw = np.asarray([debug[name] for name in names], dtype=np.float64)
        return np.sign(raw) * np.log1p(np.abs(raw)) / 10.0

    def _get_valve_diagnostics_observation(self) -> np.ndarray:
        """Return bounded privileged z/q/mdot features for a sim ablation."""
        if not self.include_valve_diagnostics:
            return np.empty(0, dtype=np.float64)
        if not hasattr(self.obs, "get_valve_debug"):
            raise RuntimeError("include_valve_diagnostics requires simulator valve diagnostics")
        debug = self.obs.get_valve_debug()
        names = (
            "ch_pos",
            "ch_neg",
            "act_pos_in",
            "act_pos_out",
            "act_neg_in",
            "act_neg_out",
        )
        values: list[float] = []
        for name in names:
            z = float(debug[f"{name}_z"])
            q = float(debug[f"{name}_q_pred_lpm"])
            mdot = float(debug[f"{name}_mdot"])
            values.extend(
                (
                    float(np.sign(z) * np.log1p(abs(z) / 1.0e4)),
                    float(np.log1p(max(q, 0.0))),
                    float(np.sign(mdot) * np.log1p(abs(mdot) / 1.0e-4)),
                )
            )
        return np.asarray(values, dtype=np.float64)

    def _get_physics_features(
        self,
        obs_traj: np.ndarray,
        ref_traj: np.ndarray,
        action_history: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Return explicit pressure-rate/reserve/action-history features."""
        if not self.include_physics_features or self.physics_feature_mode == "none":
            return np.empty(0, dtype=np.float64)
        if len(obs_traj) >= 2:
            rates = (obs_traj[-1] - obs_traj[-2]) * float(self.obs.freq)
        else:
            rates = np.zeros(4, dtype=np.float64)
        pressure = obs_traj[-1]
        reserves = np.asarray(
            [pressure[0] - pressure[2], pressure[3] - pressure[1]],
            dtype=np.float64,
        )

        # Preserve the old helper contract: rate/reserve/zero4/all do not
        # require a reference trajectory and can be used with short synthetic
        # pressure histories in scale-control tests.
        if self.physics_feature_mode == "rate":
            return rates.astype(np.float64)
        if self.physics_feature_mode == "zero4":
            # Dimension/fan-in placebo for the four rate coordinates.
            return np.zeros(4, dtype=np.float64)
        if self.physics_feature_mode == "reserve":
            return reserves.astype(np.float64)
        if self.physics_feature_mode in ("rate_reserve", "hrr"):
            return np.r_[rates, reserves].astype(np.float64)

        refs = np.asarray(ref_traj, dtype=np.float64).reshape(-1, self.dim_ref)
        if len(refs) >= self.num_obs:
            history_refs = refs[: self.num_obs, :2]
        elif len(refs) > 0:
            history_refs = np.tile(refs[-1, :2], (self.num_obs, 1))
        else:
            history_refs = np.zeros((self.num_obs, 2), dtype=np.float64)
        errors = history_refs - obs_traj[:, 2:4]
        error_rate = (errors[-1] - errors[-2]) * float(self.obs.freq)
        error_integral = np.clip(
            np.sum(errors, axis=0) / max(float(self.obs.freq), 1e-9),
            -self.integral_error_limit,
            self.integral_error_limit,
        )

        def to_physical(action: np.ndarray) -> np.ndarray:
            return float(self.obs.scale_low) + (
                float(self.obs.scale_high) - float(self.obs.scale_low)
            ) * np.asarray(action, dtype=np.float64)

        current_action, previous_action = action_history
        current_action = to_physical(current_action)
        previous_action = to_physical(previous_action)
        action_delta = current_action - previous_action
        delta_obs = obs_traj[-1] - obs_traj[-2]

        if self.physics_feature_mode == "edot":
            return error_rate.astype(np.float64)
        if self.physics_feature_mode == "rate_reserve_edot":
            return np.r_[rates, reserves, error_rate].astype(np.float64)
        if self.physics_feature_mode == "pide":
            return np.r_[errors[-1], error_integral, error_rate].astype(np.float64)
        if self.physics_feature_mode == "gpide":
            return np.r_[delta_obs, current_action, action_delta].astype(np.float64)
        return np.r_[
            rates,
            reserves,
            current_action,
            previous_action,
        ].astype(np.float64)

    def predict_obs(
        self,
        init_press: np.ndarray,
        ctrl_traj: np.ndarray,
        ref_traj: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.pred.set_init_press(
            init_press[0],
            init_press[1],
            init_press[2],
            init_press[3],
        )
        rnd_num = 2 * np.random.rand(2) - 1
        if hasattr(self.pred, "set_offset"):
            self.pred.set_offset(
                0.0,
                0.0,
                *(self.offset_range * rnd_num),
            )

        preds = np.array([], dtype=np.float64)
        pred_ctrls = []
        chamber_press = init_press[0:2]
        chamber_integral_err = self.chamber_integral_err.copy()
        for ctrl, goal in zip(ctrl_traj, ref_traj):
            pred_ctrl = ctrl.copy()
            if self.steady_chamber_ctrl is not None:
                pred_ctrl[0:2], chamber_integral_err = self._compute_steady_chamber_ctrl(
                    chamber_press,
                    chamber_integral_err,
                    integrate=True,
                )
            elif self.dynamic_chamber_ctrl is not None:
                pred_ctrl[0:2], chamber_integral_err, _ = self._compute_dynamic_chamber_ctrl(
                    chamber_press,
                    goal,
                    chamber_integral_err,
                    integrate=True,
                )
            pred, _ = self.pred.observe(pred_ctrl, goal)
            preds = np.r_[preds, pred[1:5]]
            pred_ctrls.append(pred_ctrl)
            chamber_press = pred[1:3]

        pred_press = preds.reshape(-1, self.dim_obs)
        pred_info = dict(
            pred_act=np.asarray(pred_ctrls, dtype=np.float64),
            pred_ref=ref_traj,
            pred_press=pred_press,
        )

        return pred_press, pred_info

    def _split_state(
        self,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        idx0 = self.dim_obs_traj
        obs_traj = state[:idx0].reshape(self.num_obs, self.dim_obs)

        # Action/privileged features sit between pressure history and the
        # optional predictor block.  They are not used by the reward parser.
        idx1 = idx0 + self.dim_action_history
        idx1 += self.dim_valve_z
        idx1 += self.dim_physics_features
        idx1 += self.dim_valve_diagnostics
        idx1 += self.dim_integral_error

        pred_traj = None
        if self.pred is not None:
            pred_end = idx1 + self.dim_fut_traj
            pred_traj = state[idx1:pred_end].reshape(self.num_act, self.dim_obs)
            idx1 = pred_end

        ref_traj = state[idx1:].reshape(self.num_ref, self.dim_ref)
        return obs_traj, pred_traj, ref_traj

    def _tracking_loss(self, error: float, scale: float = 1.0) -> float:
        """Return a configurable robust loss for one tracking error.

        Absolute error is the historical default.  Huber is preferable for
        the first nonlinear-plant ablation because it is quadratic near zero
        but does not make large transient errors dominate as L2 does.
        """
        mode = str(self.rwd_kwargs.get("tracking_loss_mode", "abs")).lower()
        scale = max(float(scale), 1e-9)
        x = abs(float(error)) / scale
        if mode == "square":
            return x * x
        if mode == "huber":
            delta = max(float(self.rwd_kwargs.get("tracking_huber_delta", 1.0)), 1e-9)
            return 0.5 * x * x if x <= delta else delta * (x - 0.5 * delta)
        if mode != "abs":
            raise ValueError(f"unknown tracking_loss_mode: {mode}")
        return x

    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, dict[str, float]]:
        obs_traj, pred_traj, ref_traj = self._split_state(state)

        act_obses = obs_traj[:, 2:4]
        refs = ref_traj[:self.num_obs]
        errs = refs - act_obses

        prev_errs = errs[0:-1]
        curr_err = errs[-1]

        reward = 0.0

        pos_prev_reward = np.sum(
            [self._tracking_loss(err) for err in prev_errs[:, 0]]
        )
        pos_prev_reward *= -self.rwd_kwargs["pos_prev_rwd_coeff"]
        neg_prev_reward = np.sum(
            [self._tracking_loss(err) for err in prev_errs[:, 1]]
        )
        neg_prev_reward *= -self.rwd_kwargs["neg_prev_rwd_coeff"]
        reward += pos_prev_reward + neg_prev_reward

        pos_curr_reward = self._tracking_loss(curr_err[0])
        pos_curr_reward *= -self.rwd_kwargs["pos_curr_rwd_coeff"]
        neg_curr_reward = self._tracking_loss(curr_err[1])
        neg_curr_reward *= -self.rwd_kwargs["neg_curr_rwd_coeff"]
        reward += pos_curr_reward + neg_curr_reward

        if self.pred is not None and pred_traj is not None:
            pred_act_obses = pred_traj[:, 2:4]
            pred_refs = ref_traj[self.num_obs:self.num_obs + self.num_act]
            pred_errs = pred_refs - pred_act_obses
            pred_err = pred_errs[-1]

            pos_fut_reward = np.sum(
                [self._tracking_loss(err) for err in pred_errs[:, 0]]
            )
            pos_fut_reward *= -self.rwd_kwargs["pos_fut_rwd_coeff"]
            neg_fut_reward = np.sum(
                [self._tracking_loss(err) for err in pred_errs[:, 1]]
            )
            neg_fut_reward *= -self.rwd_kwargs["neg_fut_rwd_coeff"]
            reward += pos_fut_reward + neg_fut_reward

            pos_pred_reward = self._tracking_loss(pred_err[0])
            pos_pred_reward *= -self.rwd_kwargs["pos_pred_rwd_coeff"]
            neg_pred_reward = self._tracking_loss(pred_err[1])
            neg_pred_reward *= -self.rwd_kwargs["neg_pred_rwd_coeff"]
            reward += pos_pred_reward + neg_pred_reward
        else:
            pos_fut_reward = 0.0
            neg_fut_reward = 0.0
            pos_pred_reward = 0.0
            neg_pred_reward = 0.0

        curr_ctrl = self._expand_ctrl_traj(action)[0]

        # Pressure-only error cannot distinguish "the target is rising and we
        # are late" from "the target has turned down and we are still rising".
        # This derivative term supplies the missing phase information without
        # exposing the simulator's hidden valve state to the policy.
        dt = 1.0 / max(float(self.obs.freq), 1e-9)
        actual_rate = (act_obses[-1] - self.prev_full_press[2:4]) / dt
        ref_rate = (refs[-1] - self.prev_ref_current) / dt
        rate_error = ref_rate - actual_rate
        pos_rate_loss = self._tracking_loss(
            rate_error[0], self.rwd_kwargs.get("pos_rate_scale_kpa_s", 12.0)
        )
        neg_rate_loss = self._tracking_loss(
            rate_error[1], self.rwd_kwargs.get("neg_rate_scale_kpa_s", 6.0)
        )
        pos_rate_reward = -self.rwd_kwargs.get("pos_rate_rwd_coeff", 0.0) * pos_rate_loss
        neg_rate_reward = -self.rwd_kwargs.get("neg_rate_rwd_coeff", 0.0) * neg_rate_loss
        reward += pos_rate_reward + neg_rate_reward

        # 2026-05-28:
        # lib3의 RL+PID 제어가 tracking은 되지만 밸브 동시 개방과 chamber drift로 인해
        # 거칠게 흔들리는 경향이 있어, 작은 보상 항으로 부드러움/충돌/저장압을 함께 유도한다.
        action_delta = float(np.mean((curr_ctrl - self.prev_action) ** 2))
        action_delta_reward = -self.rwd_kwargs["action_delta_rwd_coeff"] * action_delta
        reward += action_delta_reward

        second_action_delta = float(
            np.mean(
                (
                    curr_ctrl
                    - 2.0 * self.prev_action
                    + self.prev_prev_action
                )
                ** 2
            )
        )
        second_action_delta_reward = -float(
            self.rwd_kwargs.get("second_action_delta_rwd_coeff", 0.0)
        ) * second_action_delta
        reward += second_action_delta_reward

        conflict = float(
            0.5 * (
                curr_ctrl[2] * curr_ctrl[3]
                + curr_ctrl[4] * curr_ctrl[5]
            )
        )
        conflict_reward = -self.rwd_kwargs["conflict_rwd_coeff"] * conflict
        reward += conflict_reward

        chamber_margin = self.rwd_kwargs["chamber_margin_kpa"]
        chamber_deadband = self.rwd_kwargs["chamber_deadband_kpa"]
        ch_pos = obs_traj[-1, 0]
        ch_neg = obs_traj[-1, 1]
        pos_reserve_target = refs[-1, 0] + chamber_margin
        neg_reserve_target = refs[-1, 1] - chamber_margin
        tracking_channels = self.rwd_kwargs.get("tracking_channels", "both")
        track_pos = tracking_channels in ("both", "pos")
        track_neg = tracking_channels in ("both", "neg")
        pos_reserve_shortage = (
            max(pos_reserve_target - ch_pos - chamber_deadband, 0.0)
            if track_pos
            else 0.0
        )
        neg_reserve_shortage = (
            max(ch_neg - neg_reserve_target - chamber_deadband, 0.0)
            if track_neg
            else 0.0
        )
        chamber_reserve_error = pos_reserve_shortage + neg_reserve_shortage
        chamber_reserve_reward = -self.rwd_kwargs["chamber_reserve_rwd_coeff"] * chamber_reserve_error
        reward += chamber_reserve_reward

        # A one-sided reserve term rewards having enough pressure headroom but
        # does not discourage an arbitrarily over-pressurized POS chamber or
        # an over-evacuated NEG chamber.  This optional operating-envelope
        # term closes that reward loophole while leaving legacy configs
        # unchanged at coefficient zero.
        pos_bound_violation = max(
            float(self.rwd_kwargs.get("chamber_pos_min_kpa", 101.325)) - ch_pos,
            0.0,
        ) + max(
            ch_pos - float(self.rwd_kwargs.get("chamber_pos_max_kpa", 220.0)),
            0.0,
        )
        neg_bound_violation = max(
            float(self.rwd_kwargs.get("chamber_neg_min_kpa", 10.0)) - ch_neg,
            0.0,
        ) + max(
            ch_neg - float(self.rwd_kwargs.get("chamber_neg_max_kpa", 101.325)),
            0.0,
        )
        chamber_bound_error = pos_bound_violation + neg_bound_violation
        chamber_bound_reward = -float(
            self.rwd_kwargs.get("chamber_bound_rwd_coeff", 0.0)
        ) * chamber_bound_error
        reward += chamber_bound_reward

        info = {
            "pos_prev_reward": pos_prev_reward,
            "neg_prev_reward": neg_prev_reward,
            "pos_curr_reward": pos_curr_reward,
            "neg_curr_reward": neg_curr_reward,
            "pos_fut_reward": pos_fut_reward,
            "neg_fut_reward": neg_fut_reward,
            "pos_pred_reward": pos_pred_reward,
            "neg_pred_reward": neg_pred_reward,
            "pos_rate_reward": pos_rate_reward,
            "neg_rate_reward": neg_rate_reward,
            "actual_pos_rate_kpa_s": float(actual_rate[0]),
            "actual_neg_rate_kpa_s": float(actual_rate[1]),
            "ref_pos_rate_kpa_s": float(ref_rate[0]),
            "ref_neg_rate_kpa_s": float(ref_rate[1]),
            "pos_rate_error_kpa_s": float(rate_error[0]),
            "neg_rate_error_kpa_s": float(rate_error[1]),
            "action_delta_reward": action_delta_reward,
            "second_action_delta_reward": second_action_delta_reward,
            "conflict_reward": conflict_reward,
            "chamber_reserve_reward": chamber_reserve_reward,
            "chamber_bound_reward": chamber_bound_reward,
            "action_delta": action_delta,
            "second_action_delta": second_action_delta,
            "conflict": conflict,
            "chamber_reserve_error": chamber_reserve_error,
            "chamber_bound_error": chamber_bound_error,
            "chamber_pos_bound_violation": pos_bound_violation,
            "chamber_neg_bound_violation": neg_bound_violation,
        }

        return float(reward), info

    def get_info(
        self,
        state_info: Dict[str, np.ndarray],
        reward_info: Dict[str, float],
    ) -> Dict[str, Union[np.ndarray, float]]:
        return dict(
            obs=state_info["Observation"],
            obs_wo_noise=state_info["obs_w/o_noise"],
            ctrl_input=state_info["ctrl"],
            pred=state_info["pred"],
            goal_traj=state_info["goal_traj"],
            full_press=state_info["full_press"],
            chamber_ctrl=state_info.get("chamber_ctrl", np.array([np.nan, np.nan])),
            chamber_target=state_info.get("chamber_target", np.array([np.nan, np.nan])),
            valve_debug=state_info.get("valve_debug", {}),
            reward=reward_info,
        )

    def verbose(self, info: Dict[str, Any]) -> None:
        applied_ctrl = np.array(
            [
                info["obs"]["ctrl_pos"],
                info["obs"]["ctrl_neg"],
                info["obs"]["ctrl_act_pos_in"],
                info["obs"]["ctrl_act_pos_out"],
                info["obs"]["ctrl_act_neg_in"],
                info["obs"]["ctrl_act_neg_out"],
            ],
            dtype=np.float64,
        )
        raw_ctrl = np.asarray(info["ctrl_input"], dtype=np.float64).reshape(-1, self.dim_act)[0]
        ctrl_str = np.array2string(
            applied_ctrl,
            precision=4,
            max_line_width=1000,
            separator=" ",
        )
        raw_ctrl_str = np.array2string(
            raw_ctrl,
            precision=4,
            max_line_width=1000,
            separator=" ",
        )
        obs_wo_noise_str = np.array2string(
            np.asarray(info["obs_wo_noise"], dtype=np.float64),
            precision=4,
            max_line_width=1000,
            separator=" ",
        )
        lines = [
            '[ INFO] Pneumatic Env3 ==>',
            f'\tTime: {info["obs"]["curr_time"]}',
            f'\tCh  : (\t{info["obs"]["sen_pos"]:3.4f}\t{info["obs"]["sen_neg"]:3.4f})',
            f'\tAct : (\t{info["obs"]["sen_act_pos"]:3.4f}\t{info["obs"]["sen_act_neg"]:3.4f})',
            f'\tRef : (\t{info["obs"]["ref_act_pos"]:3.4f}\t{info["obs"]["ref_act_neg"]:3.4f})',
            f'\tCtrl: {ctrl_str}',
            f'\tC/I : {raw_ctrl_str}',
            f'\tw/o : {obs_wo_noise_str}',
            (
                f'\tRWD : Curr \t'
                f'{info["reward"]["pos_curr_reward"]:.4f}\t'
                f'{info["reward"]["neg_curr_reward"]:.4f}'
            ),
            (
                f'\t    : Prev \t'
                f'{info["reward"]["pos_prev_reward"]:.4f}\t'
                f'{info["reward"]["neg_prev_reward"]:.4f}'
            ),
            (
                f'\t    : Fut  \t'
                f'{info["reward"]["pos_fut_reward"]:.4f}\t'
                f'{info["reward"]["neg_fut_reward"]:.4f}'
            ),
            (
                f'\t    : Pred \t'
                f'{info["reward"]["pos_pred_reward"]:.4f}\t'
                f'{info["reward"]["neg_pred_reward"]:.4f}'
            ),
            (
                f'\t    : Delta\t'
                f'{info["reward"]["action_delta_reward"]:.4f}\t'
                f'val={info["reward"]["action_delta"]:.4f}'
            ),
            (
                f'\t    : Conf \t'
                f'{info["reward"]["conflict_reward"]:.4f}\t'
                f'val={info["reward"]["conflict"]:.4f}'
            ),
            (
                f'\t    : ChRes\t'
                f'{info["reward"]["chamber_reserve_reward"]:.4f}\t'
                f'err={info["reward"]["chamber_reserve_error"]:.4f}'
            ),
            f'\t    : Total\t{sum(value for key, value in info["reward"].items() if key.endswith("_reward")):.4f}',
        ]
        output = "\n".join(lines)
        print(output)
        for _ in range(len(lines)):
            sys.stdout.write("\x1b[1A")
            sys.stdout.write("\x1b[2K")

    def set_volume(self, vol1: float, vol2: float) -> None:
        self.obs.set_volume(vol1, vol2)
        if self.pred is not None:
            self.pred.set_volume(vol1, vol2)

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
        Ka: Optional[float] = None,
    ) -> None:
        self.is_pid = True
        self.obs.set_pid(
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
        )
        if Ka is not None:
            self.obs.set_anti_windup(Ka)

    def publish_observation(self, info: Dict[str, Any]) -> None:
        raise NotImplementedError("PneuEnv3 observation publishing is not implemented.")


if __name__ == "__main__":
    from pneu_ref.random_ref import RandomRef
    from pneu_env.sim3 import PneuSim
    from pneu_env.pred3 import PneuPred

    obs = PneuSim()
    ref = RandomRef()
    pred = PneuPred()
    env = PneuEnv3(obs=obs, ref=ref, pred=pred, verbose=False)
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    state, info = env.reset()
    print(state.shape)
    print(env.step(action))
