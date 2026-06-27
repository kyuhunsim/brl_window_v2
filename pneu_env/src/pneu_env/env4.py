import sys

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from gymnasium.spaces import Box

from pneu_ref.pneu_ref_nd import PneuRefND


class PneuEnv4:
    def __init__(
        self,
        obs: Any,
        ref: Any,
        pred: Any = None,
        num_prev: int = 10,
        num_pred: int = 10,
        num_act: int = 1,
        rwd_kwargs: Dict[str, float] = dict(
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_curr_rwd_coeff=0.0,
            neg_curr_rwd_coeff=0.0,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
            diff_curr_rwd_coeff=0.0,
            diff_fut_rwd_coeff=0.0,
            diff_pred_rwd_coeff=0.0,
            disp_prev_rwd_coeff=0.0,
            disp_curr_rwd_coeff=0.0,
            disp_fut_rwd_coeff=0.0,
            disp_pred_rwd_coeff=0.0,
            disp_vel_rwd_coeff=0.0,
            action_delta_rwd_coeff=0.0,
        ),
        pos_pred_rnd_offset_range: float = 0,
        neg_pred_rnd_offset_range: float = 0,
        verbose: bool = True,
        action_low: float = 0.0,
        action_high: float = 1.0,
        episode_carry_state: bool = True,
        fixed_chamber_ctrl: Optional[Tuple[float, float]] = None,
        steady_chamber_ctrl: Optional[Dict[str, float]] = None,
        displacement_obs_key: str = "sen_total_contraction_from_rest",
        displacement_scale: float = 1000.0,
        velocity_scale: float = 1000.0,
        include_displacement_velocity_obs: bool = True,
        include_prev_action_obs: bool = False,
        include_pressure_diff_velocity_obs: bool = False,
    ):
        self.obs = obs
        self.goal = PneuRefND(
            ref,
            num_prev=num_prev,
            num_pred=num_pred,
            ctrl_freq=self.obs.freq,
        )
        self.pred = pred

        self.num_prev = num_prev
        self.num_pred = num_pred
        self.num_act = num_act

        self.num_obs = num_prev + 1
        self.num_ref = num_prev + num_pred + 1

        self.include_displacement_velocity_obs = bool(include_displacement_velocity_obs)
        self.press_idx = slice(0, 4)
        self.disp_idx = 4
        self.vel_idx = 5 if self.include_displacement_velocity_obs else None
        self.base_obs_dim = 5 + (1 if self.include_displacement_velocity_obs else 0)
        self.dim_ref = 3
        self.dim_act = 6
        self.include_prev_action_obs = bool(include_prev_action_obs)
        self.include_pressure_diff_velocity_obs = bool(include_pressure_diff_velocity_obs)
        self.prev_action_obs_dim = self.dim_act if self.include_prev_action_obs else 0
        self.pressure_diff_velocity_obs_dim = 1 if self.include_pressure_diff_velocity_obs else 0
        self.dim_obs = (
            self.base_obs_dim
            + self.prev_action_obs_dim
            + self.pressure_diff_velocity_obs_dim
        )
        if self.goal.dim_ref != self.dim_ref:
            raise ValueError(
                f"PneuEnv4 expects 3D refs [act_pos, act_neg, displacement], "
                f"got dim {self.goal.dim_ref}"
            )
        self.displacement_obs_key = displacement_obs_key
        self.displacement_scale = float(displacement_scale)
        self.velocity_scale = float(velocity_scale)
        self.fixed_chamber_ctrl = None
        self.steady_chamber_ctrl = None
        if fixed_chamber_ctrl is not None and steady_chamber_ctrl is not None:
            raise ValueError("fixed_chamber_ctrl and steady_chamber_ctrl are mutually exclusive")
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
        self.dim_policy_act = (
            4
            if self.fixed_chamber_ctrl is not None or self.steady_chamber_ctrl is not None
            else self.dim_act
        )

        self.dim_obs_traj = self.num_obs * self.dim_obs
        self.dim_fut_traj = self.num_act * self.dim_obs if pred is not None else 0
        self.dim_ref_traj = self.num_ref * self.dim_ref
        self.dim_act_traj = self.num_act * self.dim_policy_act
        self.dim_state = self.dim_obs_traj + self.dim_fut_traj + self.dim_ref_traj

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

        init_obs = self._initial_obs_vector()
        self.obs_traj = np.tile(init_obs, (self.num_obs, 1))
        self.curr_full_press = init_obs.copy()
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
        self.has_episode_state = False
        self.chamber_integral_err = np.zeros(2, dtype=np.float64)
        self.prev_action = self._expand_ctrl_traj(self.action_space.low)[0]

    def _initial_obs_vector(self) -> np.ndarray:
        base_obs = np.array(
            [
                getattr(self.obs, "init_pos_press", 101.325),
                getattr(self.obs, "init_neg_press", 101.325),
                getattr(self.obs, "init_act_pos_press", 101.325),
                getattr(self.obs, "init_act_neg_press", 101.325),
                self._sim_cell_length_to_ref_disp(
                    getattr(self.obs, "init_length", getattr(self.obs, "initial_length", 0.0)),
                    self.obs,
                ),
            ],
            dtype=np.float64,
        )
        if self.include_displacement_velocity_obs:
            base_obs = np.r_[base_obs, 0.0]
        return self._augment_obs(
            base_obs,
            prev_base_obs=None,
            applied_ctrl=np.zeros(self.dim_act, dtype=np.float64),
        )

    def _augment_obs(
        self,
        base_obs: np.ndarray,
        prev_base_obs: Optional[np.ndarray],
        applied_ctrl: Optional[np.ndarray],
    ) -> np.ndarray:
        components = [np.asarray(base_obs, dtype=np.float64)]

        if self.include_prev_action_obs:
            if applied_ctrl is None:
                applied_ctrl = np.zeros(self.dim_act, dtype=np.float64)
            components.append(np.asarray(applied_ctrl, dtype=np.float64).reshape(self.dim_act))

        if self.include_pressure_diff_velocity_obs:
            if prev_base_obs is None:
                diff_velocity = 0.0
            else:
                curr_diff = float(base_obs[2] - base_obs[3])
                prev_diff = float(prev_base_obs[2] - prev_base_obs[3])
                diff_velocity = (curr_diff - prev_diff) * float(self.obs.freq)
            components.append(np.array([diff_velocity], dtype=np.float64))

        return np.concatenate(components, dtype=np.float64)

    def _sim_cell_length_to_ref_disp(self, cell_length: float, sim: Any) -> float:
        if self.displacement_obs_key == "sen_total_length":
            value_m = sim.cell_length_to_total_length(cell_length)
        elif self.displacement_obs_key == "sen_total_contraction":
            value_m = sim.cell_to_total_displacement(getattr(sim, "init_length", cell_length) - cell_length)
        else:
            total_length = sim.cell_length_to_total_length(cell_length)
            value_m = sim.total_length_to_contraction(total_length)
        return float(value_m) * self.displacement_scale

    def _sim_obs_to_base_obs(
        self,
        obs: np.ndarray,
        obs_info: Dict[str, Any],
    ) -> np.ndarray:
        observation = obs_info.get("Observation", {})
        if self.displacement_obs_key in observation:
            displacement_m = float(observation[self.displacement_obs_key])
        else:
            displacement_m = self._sim_cell_length_to_ref_disp(obs[5], self.obs) / self.displacement_scale

        components = [
            obs[1],
            obs[2],
            obs[3],
            obs[4],
            displacement_m * self.displacement_scale,
        ]

        if self.include_displacement_velocity_obs:
            total_velocity_m_s = float(observation.get("sen_total_velocity", obs[6]))
            if "contraction" in self.displacement_obs_key:
                displacement_velocity_m_s = -total_velocity_m_s
            else:
                displacement_velocity_m_s = total_velocity_m_s
            components.append(displacement_velocity_m_s * self.velocity_scale)

        return np.array(components, dtype=np.float64)

    def _extract_applied_ctrl(self, obs_info: Dict[str, Any]) -> np.ndarray:
        observation = obs_info.get("Observation", {})
        return np.array(
            [
                observation["ctrl_pos"],
                observation["ctrl_neg"],
                observation["ctrl_act_pos_in"],
                observation["ctrl_act_pos_out"],
                observation["ctrl_act_neg_in"],
                observation["ctrl_act_neg_out"],
            ],
            dtype=np.float64,
        )

    def _env_motion_to_sim_state(
        self,
        motion_obs: np.ndarray,
        sim: Any,
    ) -> Tuple[float, float]:
        displacement_m = float(motion_obs[0]) / self.displacement_scale
        if self.include_displacement_velocity_obs and len(motion_obs) > 1:
            displacement_velocity_m_s = float(motion_obs[1]) / self.velocity_scale
        else:
            displacement_velocity_m_s = 0.0

        if self.displacement_obs_key == "sen_total_length":
            total_length = displacement_m
            total_velocity_m_s = displacement_velocity_m_s
        elif self.displacement_obs_key == "sen_total_contraction":
            cell_length = getattr(sim, "init_length", getattr(sim, "initial_length", 0.0))
            cell_length -= sim.total_to_cell_displacement(displacement_m)
            cell_velocity_m_s = -sim.total_to_cell_displacement(displacement_velocity_m_s)
            return sim.clamp_cell_length(cell_length), float(cell_velocity_m_s)
        else:
            total_length = sim.actuator_total_initial_length - displacement_m
            total_velocity_m_s = -displacement_velocity_m_s

        cell_length = sim.total_length_to_cell_length(total_length)
        cell_velocity_m_s = sim.total_to_cell_displacement(total_velocity_m_s)
        return sim.clamp_cell_length(cell_length), float(cell_velocity_m_s)

    def _expand_ctrl_traj(
        self,
        ctrl: np.ndarray,
        update_chamber_integral: bool = False,
    ) -> np.ndarray:
        ctrl_traj = np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_policy_act)
        if self.fixed_chamber_ctrl is None and self.steady_chamber_ctrl is None:
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

    def _get_chamber_ctrl(
        self,
        chamber_press: Optional[np.ndarray] = None,
        update_integral: bool = False,
    ) -> np.ndarray:
        if self.fixed_chamber_ctrl is not None:
            return self.fixed_chamber_ctrl.copy()
        if self.steady_chamber_ctrl is None:
            raise RuntimeError("chamber controller is not configured")

        if chamber_press is None:
            chamber_press = self.curr_full_press[0:2]
        ctrl, integral_err = self._compute_steady_chamber_ctrl(
            chamber_press,
            self.chamber_integral_err,
            integrate=update_integral,
        )
        if update_integral:
            self.chamber_integral_err = integral_err
        return ctrl

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_init_press = self._initial_obs_vector()
        if self.episode_carry_state and self.has_episode_state:
            init_press = self.curr_full_press.copy()
        else:
            init_press = base_init_press
        if hasattr(self.obs, "set_init_press"):
            init_length, init_velocity = self._env_motion_to_sim_state(
                init_press[self.disp_idx:self.base_obs_dim],
                self.obs,
            )
            self.obs.set_init_press(
                init_press[0],
                init_press[1],
                init_press[2],
                init_press[3],
                init_length,
                init_velocity,
            )
        if hasattr(self.goal, "reset"):
            self.goal.reset()
        self.obs_traj = np.tile(init_press, (self.num_obs, 1))
        self.curr_full_press = init_press.copy()
        self.t = 0.0
        self.chamber_integral_err = np.zeros(2, dtype=np.float64)
        self.prev_action = self._expand_ctrl_traj(self.action_space.low)[0]

        if self.is_pid:
            self.obs.reset_pid()

        ctrl = np.asarray(self.action_space.low, dtype=np.float64)
        state, _, _, _, info = self.step(ctrl)
        self.t = info["obs"]["curr_time"]

        if self.is_pid:
            self.obs.reset_pid()

        return state, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state, state_info = self.get_state(action)
        reward, reward_info = self.get_reward(
            state=state,
            action=action,
        )
        terminated = False
        truncated = False

        info = self.get_info(state_info, reward_info)
        if self.verbose_enabled:
            self.verbose(info)
        if self.publish_obs:
            self.publish_observation(info)

        self.prev_action = np.asarray(state_info["ctrl"], dtype=np.float64)[0].copy()

        return state, reward, terminated, truncated, info

    def close(self) -> None:
        ctrl = np.zeros(self.dim_act, dtype=np.float64)
        goal = np.array([101.325, 101.325, 0.0], dtype=np.float64)
        _, _ = self.obs.observe(ctrl, goal)

    def get_state(
        self,
        ctrl: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl_traj = self._expand_ctrl_traj(ctrl, update_chamber_integral=True)
        goal_traj = self.goal.get_ref(self.t).reshape(-1, self.dim_ref)
        prev_base_obs = self.obs_traj[-1, :self.base_obs_dim].copy()

        obs, obs_info = self.obs.observe(
            ctrl_traj.copy()[0],
            goal_traj.copy()[self.num_obs - 1],
        )
        base_obs = self._sim_obs_to_base_obs(obs, obs_info)
        applied_ctrl = self._extract_applied_ctrl(obs_info)
        full_press = self._augment_obs(base_obs, prev_base_obs, applied_ctrl).reshape(1, self.dim_obs)
        obs_traj = np.r_[
            self.obs_traj[1:],
            full_press,
        ]

        pred_info = None
        if self.pred is not None:
            pred_traj, pred_info = self.predict_obs(
                init_press=obs_traj[-1],
                ctrl_traj=ctrl_traj,
                ref_traj=goal_traj[self.num_obs:self.num_obs + self.num_act],
            )
            state = np.r_[
                obs_traj.reshape(-1),
                pred_traj.reshape(-1),
                goal_traj.reshape(-1),
            ]
        else:
            state = np.r_[
                obs_traj.reshape(-1),
                goal_traj.reshape(-1),
            ]

        self.t = obs[0]
        self.obs_traj = obs_traj
        self.curr_full_press = obs_traj[-1].copy()
        self.has_episode_state = True
        obs_info["Observation"]["sen_env_displacement"] = float(full_press[0, 4])
        obs_info["Observation"]["sen_env_velocity"] = (
            float(full_press[0, self.vel_idx]) if self.vel_idx is not None else 0.0
        )
        if self.include_pressure_diff_velocity_obs:
            obs_info["Observation"]["sen_pressure_diff_velocity"] = float(full_press[0, -1])
        obs_info["ctrl"] = ctrl_traj
        obs_info["pred"] = pred_info
        obs_info["goal_traj"] = goal_traj
        obs_info["full_press"] = self.curr_full_press.copy()

        return state, obs_info

    def predict_obs(
        self,
        init_press: np.ndarray,
        ctrl_traj: np.ndarray,
        ref_traj: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        init_base_obs = init_press[:self.base_obs_dim]
        init_length, init_velocity = self._env_motion_to_sim_state(
            init_base_obs[self.disp_idx:self.base_obs_dim],
            self.pred,
        )
        self.pred.set_init_press(
            init_base_obs[0],
            init_base_obs[1],
            init_base_obs[2],
            init_base_obs[3],
            init_length,
            init_velocity,
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
        chamber_press = init_base_obs[0:2]
        chamber_integral_err = self.chamber_integral_err.copy()
        prev_base_obs = init_base_obs.copy()
        for ctrl, goal in zip(ctrl_traj, ref_traj):
            pred_ctrl = ctrl.copy()
            if self.steady_chamber_ctrl is not None:
                pred_ctrl[0:2], chamber_integral_err = self._compute_steady_chamber_ctrl(
                    chamber_press,
                    chamber_integral_err,
                    integrate=True,
                )
            pred, pred_obs_info = self.pred.observe(pred_ctrl, goal)
            pred_base_obs = self._sim_obs_to_base_obs(pred, pred_obs_info)
            pred_applied_ctrl = self._extract_applied_ctrl(pred_obs_info)
            pred_env_obs = self._augment_obs(pred_base_obs, prev_base_obs, pred_applied_ctrl)
            preds = np.r_[preds, pred_env_obs]
            pred_ctrls.append(pred_ctrl)
            chamber_press = pred_base_obs[0:2]
            prev_base_obs = pred_base_obs

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

        pred_traj = None
        idx1 = idx0
        if self.pred is not None:
            idx1 = idx0 + self.dim_fut_traj
            pred_traj = state[idx0:idx1].reshape(self.num_act, self.dim_obs)

        ref_traj = state[idx1:].reshape(self.num_ref, self.dim_ref)
        return obs_traj, pred_traj, ref_traj

    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, dict[str, float]]:
        obs_traj, pred_traj, ref_traj = self._split_state(state)

        act_obses = obs_traj[:, 2:4]
        disp_obses = obs_traj[:, self.disp_idx]
        refs = ref_traj[:self.num_obs]
        press_refs = refs[:, 0:2]
        disp_refs = refs[:, 2]
        errs = press_refs - act_obses
        disp_errs = disp_refs - disp_obses
        diff_refs = press_refs[:, 0] - press_refs[:, 1]
        diff_obses = act_obses[:, 0] - act_obses[:, 1]
        diff_errs = diff_refs - diff_obses

        prev_errs = errs[0:-1]
        curr_err = errs[-1]
        prev_disp_errs = disp_errs[0:-1]
        curr_disp_err = disp_errs[-1]
        curr_diff_err = diff_errs[-1]

        reward = 0.0

        pos_prev_reward = np.sum(np.abs(prev_errs[:, 0]))
        pos_prev_reward *= -self.rwd_kwargs["pos_prev_rwd_coeff"]
        neg_prev_reward = np.sum(np.abs(prev_errs[:, 1]))
        neg_prev_reward *= -self.rwd_kwargs["neg_prev_rwd_coeff"]
        reward += pos_prev_reward + neg_prev_reward

        pos_curr_reward = np.abs(curr_err[0])
        pos_curr_reward *= -self.rwd_kwargs["pos_curr_rwd_coeff"]
        neg_curr_reward = np.abs(curr_err[1])
        neg_curr_reward *= -self.rwd_kwargs["neg_curr_rwd_coeff"]
        reward += pos_curr_reward + neg_curr_reward

        disp_prev_reward = np.sum(np.abs(prev_disp_errs))
        disp_prev_reward *= -self.rwd_kwargs["disp_prev_rwd_coeff"]
        disp_curr_reward = np.abs(curr_disp_err)
        disp_curr_reward *= -self.rwd_kwargs["disp_curr_rwd_coeff"]
        if self.vel_idx is not None:
            disp_vel_reward = np.abs(obs_traj[-1, self.vel_idx])
            disp_vel_reward *= -self.rwd_kwargs["disp_vel_rwd_coeff"]
        else:
            disp_vel_reward = 0.0
        diff_curr_reward = np.abs(curr_diff_err)
        diff_curr_reward *= -self.rwd_kwargs["diff_curr_rwd_coeff"]
        curr_action = self._expand_ctrl_traj(action)[0]
        action_delta = float(np.mean((curr_action - self.prev_action) ** 2))
        action_delta_coeff = float(self.rwd_kwargs.get("action_delta_rwd_coeff", 0.0))
        action_delta_reward = -action_delta_coeff * action_delta
        reward += (
            disp_prev_reward
            + disp_curr_reward
            + disp_vel_reward
            + diff_curr_reward
            + action_delta_reward
        )

        if self.pred is not None and pred_traj is not None:
            pred_act_obses = pred_traj[:, 2:4]
            pred_disp_obses = pred_traj[:, self.disp_idx]
            pred_refs = ref_traj[self.num_obs:self.num_obs + self.num_act]
            pred_press_refs = pred_refs[:, 0:2]
            pred_disp_refs = pred_refs[:, 2]
            pred_errs = pred_press_refs - pred_act_obses
            pred_disp_errs = pred_disp_refs - pred_disp_obses
            pred_diff_refs = pred_press_refs[:, 0] - pred_press_refs[:, 1]
            pred_diff_obses = pred_act_obses[:, 0] - pred_act_obses[:, 1]
            pred_diff_errs = pred_diff_refs - pred_diff_obses
            pred_err = pred_errs[-1]
            pred_disp_err = pred_disp_errs[-1]
            pred_diff_err = pred_diff_errs[-1]

            pos_fut_reward = np.sum(np.abs(pred_errs[:, 0]))
            pos_fut_reward *= -self.rwd_kwargs["pos_fut_rwd_coeff"]
            neg_fut_reward = np.sum(np.abs(pred_errs[:, 1]))
            neg_fut_reward *= -self.rwd_kwargs["neg_fut_rwd_coeff"]
            reward += pos_fut_reward + neg_fut_reward

            pos_pred_reward = np.abs(pred_err[0])
            pos_pred_reward *= -self.rwd_kwargs["pos_pred_rwd_coeff"]
            neg_pred_reward = np.abs(pred_err[1])
            neg_pred_reward *= -self.rwd_kwargs["neg_pred_rwd_coeff"]
            reward += pos_pred_reward + neg_pred_reward

            disp_fut_reward = np.sum(np.abs(pred_disp_errs))
            disp_fut_reward *= -self.rwd_kwargs["disp_fut_rwd_coeff"]
            disp_pred_reward = np.abs(pred_disp_err)
            disp_pred_reward *= -self.rwd_kwargs["disp_pred_rwd_coeff"]
            diff_fut_reward = np.sum(np.abs(pred_diff_errs))
            diff_fut_reward *= -self.rwd_kwargs["diff_fut_rwd_coeff"]
            diff_pred_reward = np.abs(pred_diff_err)
            diff_pred_reward *= -self.rwd_kwargs["diff_pred_rwd_coeff"]
            reward += disp_fut_reward + disp_pred_reward + diff_fut_reward + diff_pred_reward
        else:
            pos_fut_reward = 0.0
            neg_fut_reward = 0.0
            pos_pred_reward = 0.0
            neg_pred_reward = 0.0
            disp_fut_reward = 0.0
            disp_pred_reward = 0.0
            diff_fut_reward = 0.0
            diff_pred_reward = 0.0

        info = {
            "pos_prev_reward": pos_prev_reward,
            "neg_prev_reward": neg_prev_reward,
            "pos_curr_reward": pos_curr_reward,
            "neg_curr_reward": neg_curr_reward,
            "pos_fut_reward": pos_fut_reward,
            "neg_fut_reward": neg_fut_reward,
            "pos_pred_reward": pos_pred_reward,
            "neg_pred_reward": neg_pred_reward,
            "diff_curr_reward": diff_curr_reward,
            "diff_fut_reward": diff_fut_reward,
            "diff_pred_reward": diff_pred_reward,
            "disp_prev_reward": disp_prev_reward,
            "disp_curr_reward": disp_curr_reward,
            "disp_fut_reward": disp_fut_reward,
            "disp_pred_reward": disp_pred_reward,
            "disp_vel_reward": disp_vel_reward,
            "action_delta_reward": action_delta_reward,
            "action_delta": action_delta,
            "diff_error": float(curr_diff_err),
            "disp_error": float(curr_disp_err),
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
            '[ INFO] Pneumatic Env4 ==>',
            f'\tTime: {info["obs"]["curr_time"]}',
            f'\tCh  : (\t{info["obs"]["sen_pos"]:3.4f}\t{info["obs"]["sen_neg"]:3.4f})',
            f'\tAct : (\t{info["obs"]["sen_act_pos"]:3.4f}\t{info["obs"]["sen_act_neg"]:3.4f})',
            (
                f'\tDisp: {info["obs"].get("sen_env_displacement", 0.0):3.4f} mm '
                f'({info["obs"].get("sen_env_velocity", 0.0):3.4f} mm/s)'
            ),
            (
                f'\tRef : (\t{info["obs"]["ref_act_pos"]:3.4f}\t'
                f'{info["obs"]["ref_act_neg"]:3.4f}\t'
                f'{info["obs"].get("ref_displacement", 0.0):3.4f})'
            ),
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
                f'\t    : Disp \t'
                f'{info["reward"]["disp_curr_reward"]:.4f}\t'
                f'err={info["reward"]["disp_error"]:.4f}'
            ),
            (
                f'\t    : Diff \t'
                f'{info["reward"]["diff_curr_reward"]:.4f}\t'
                f'err={info["reward"]["diff_error"]:.4f}'
            ),
            (
                f'\t    : ActD \t'
                f'{info["reward"]["action_delta_reward"]:.4f}\t'
                f'delta={info["reward"]["action_delta"]:.4f}'
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
        raise NotImplementedError("PneuEnv4 observation publishing is not implemented.")


if __name__ == "__main__":
    from pneu_ref.random_ref import RandomRef
    from pneu_ref.disp_ref import PressureDisplacementRef, RandomDispRef
    from pneu_env.sim4 import PneuSim
    from pneu_env.pred4 import PneuPred

    obs = PneuSim()
    ref = PressureDisplacementRef(RandomRef(), RandomDispRef())
    pred = PneuPred()
    env = PneuEnv4(obs=obs, ref=ref, pred=pred, verbose=False)
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    state, info = env.reset()
    print(state.shape)
    print(env.step(action))
