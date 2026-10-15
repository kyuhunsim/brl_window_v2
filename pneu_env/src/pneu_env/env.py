import sys

import numpy as np
from typing import Tuple, Dict, Any, Union, Optional

from gymnasium.spaces import Box

from pneu_ref.pneu_ref import PneuRef
from pneu_env.pid import PID

class PneuEnv():
    def __init__(
        self,
        obs: Any,
        ref: Any,
        pred: Any = None,
        num_prev: int = 10,
        num_pred: int = 10,
        num_act: int = 1,
        rwd_kwargs: Dict[str, float] = dict(
            pos_prev_rwd_coeff = 0.0,
            neg_prev_rwd_coeff = 0.0,
            pos_curr_rwd_coeff = 0.0,
            neg_curr_rwd_coeff = 0.0,
            pos_fut_rwd_coeff = 0.0,
            neg_fut_rwd_coeff = 0.0,
            pos_pred_rwd_coeff = 0.0,
            neg_pred_rwd_coeff = 0.0,
        ),
        pos_pred_rnd_offset_range: float = 0,
        neg_pred_rnd_offset_range: float = 0,
        action_low: float = 0.0,
        action_high: float = 1.0,
        verbose: bool = True,
        episode_carry_state: bool = False,
        verbose_clear: bool = True,
        reference_reset_to_observation: bool = True,
        action_history_len: int = 0,
        include_physics_features: bool = False,
        physics_feature_mode: str = "rate",
        action_chunk_steps: int = 1,
        action_chunk_gamma: float = 0.9,
    ):
        self.obs = obs
        self.goal = PneuRef(
            ref,
            num_prev = num_prev,
            num_pred = num_pred,
            ctrl_freq = self.obs.freq
        )
        self.pred = pred
        
        # Definition of Env dimensions
        self.num_prev = num_prev
        self.num_pred = num_pred
        self.num_act = num_act
        
        self.num_obs = num_prev + 1
        self.num_ref = num_prev + num_pred + 1

        self.dim_obs = 2
        self.dim_act = 2

        self.action_history_len = max(int(action_history_len), 0)
        self.dim_action_history = self.action_history_len * self.dim_act
        self.include_physics_features = bool(include_physics_features)
        self.physics_feature_mode = str(physics_feature_mode).lower()
        physics_dims = {"none": 0, "rate": 2}
        if self.physics_feature_mode not in physics_dims:
            raise ValueError(
                "lib2 physics_feature_mode must be 'none' or 'rate'"
            )
        self.dim_physics_features = (
            physics_dims[self.physics_feature_mode]
            if self.include_physics_features
            else 0
        )
        self.action_chunk_steps = max(int(action_chunk_steps), 1)
        self.action_chunk_gamma = float(action_chunk_gamma)
        if not 0.0 < self.action_chunk_gamma <= 1.0:
            raise ValueError("action_chunk_gamma must lie in (0, 1]")

        self.dim_obs_traj = self.num_obs*self.dim_obs
        self.dim_fut_traj = self.num_act*self.dim_obs if pred is not None else 0
        self.dim_ref_traj = self.num_ref*self.dim_obs
        self.action_output_steps = (
            self.action_chunk_steps if self.action_chunk_steps > 1 else self.num_act
        )
        self.dim_act_traj = self.action_output_steps*self.dim_act
        self.dim_state = (
            self.dim_obs_traj
            + self.dim_action_history
            + self.dim_physics_features
            + self.dim_fut_traj
            + self.dim_ref_traj
        )
        
        # Env dimension
        self.observation_space = Box(
            low = -np.inf,
            high = np.inf,
            shape = (self.dim_state, ),
            dtype = np.float64
        )
        self.action_space = Box(
            low = action_low,
            high = action_high,
            shape = (self.dim_act_traj,),
            dtype = np.float64
        )

        # Initialize parameters
        self.publish_obs = False
        self.verbose_enabled = bool(verbose)
        self.verbose_clear = bool(verbose_clear)
        self.episode_carry_state = bool(episode_carry_state)
        self.reference_reset_to_observation = bool(reference_reset_to_observation)
        self.has_episode_state = False
        self.obs_traj = 101.325*np.ones((self.num_obs, self.dim_obs), dtype=np.float64)
        self.t = 0.0
        self.rwd_kwargs = rwd_kwargs
        self.offset_range = np.array([
            pos_pred_rnd_offset_range,
            neg_pred_rnd_offset_range
        ], dtype=np.float64)

        self.is_pid = False
        self.curr_obs = 101.325*np.ones((self.dim_obs), dtype=np.float32)
        self.curr_ref = 101.325*np.ones((self.dim_obs), dtype=np.float32)
        self.last_action = np.asarray(self.action_space.low, dtype=np.float64).copy()
        self.action_history = np.tile(
            self.action_space.low[:self.dim_act],
            (self.action_history_len, 1),
        )
        self.policy_state_scale = np.r_[
            np.full(self.dim_obs_traj, 100.0, dtype=np.float64),
            np.ones(self.dim_action_history, dtype=np.float64),
            np.full(self.dim_physics_features, 50.0, dtype=np.float64),
            np.full(self.dim_fut_traj, 100.0, dtype=np.float64),
            np.full(self.dim_ref_traj, 100.0, dtype=np.float64),
        ]
        if self.policy_state_scale.shape != (self.dim_state,):
            raise RuntimeError(
                "lib2 policy-state scale does not match observation dimension: "
                f"{self.policy_state_scale.shape} != {(self.dim_state,)}"
            )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        # With carry enabled, an episode boundary is only an RL bookkeeping
        # boundary.  Do not reinitialize the pneumatic simulator: doing so
        # would discard pump/valve internal states and would also inject an
        # unrecorded action through the reset step below.  The reference clock
        # is intentionally continuous as well, so a sequence of horizons is
        # equivalent to one long physical rollout.
        if self.episode_carry_state and self.has_episode_state:
            state, state_info = self._state_from_current(self.last_action)
            return state, self.get_info(state_info, {})

        base_init_obs = np.array([
            getattr(self.obs, "init_pos_press", 101.325),
            getattr(self.obs, "init_neg_press", 101.325),
        ], dtype=np.float64)
        if self.episode_carry_state and self.has_episode_state:
            init_obs = self.curr_obs.astype(np.float64, copy=True)
        else:
            init_obs = base_init_obs
        if hasattr(self.obs, "set_init_press"):
            self.obs.set_init_press(init_obs[0], init_obs[1])
        if hasattr(self.goal, "reset"):
            # Physical initialization and reference initialization are
            # separate experimental choices.  The historical lib2 task starts
            # the plant at atmosphere but asks it to move toward the operating
            # pressure band immediately.  Only override the reference when
            # explicitly requested by the environment configuration.
            if self.reference_reset_to_observation:
                self.goal.reset(initial_state=init_obs)
            else:
                self.goal.reset()
        self.obs_traj = np.tile(init_obs, (self.num_obs, 1))
        self.curr_obs = init_obs.astype(np.float32, copy=True)
        self.t = 0.0
        self.last_action = np.asarray(self.action_space.low, dtype=np.float64).copy()
        self.action_history = np.tile(
            self.action_space.low[:self.dim_act],
            (self.action_history_len, 1),
        )

        if self.is_pid:
            self.obs.reset_pid()

        # Establish the initial policy state without advancing the plant.
        # Calling step(action_space.low) here injected an unrecorded physical
        # command at every reset; in lib2 that command can already move the
        # negative pressure because the lower action range is in the valve
        # cracking region.
        state, state_info = self._state_from_current(self.last_action)
        return state, self.get_info(state_info, {})
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        if self.action_chunk_steps > 1:
            state, state_info, reward, reward_info = self._get_chunk_state(action)
        else:
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

        return (
            state,
            reward,
            terminated,
            truncated,
            info
        )
        
    def close(self):
        ctrl = np.ones(self.dim_act, dtype=np.float64)
        goal = 101.325*np.ones(self.dim_obs, dtype=np.float64)
        _, _ = self.obs.observe(ctrl, goal)

    def _append_action_history(self, action: np.ndarray) -> np.ndarray:
        if self.action_history_len == 0:
            return np.empty((0, self.dim_act), dtype=np.float64)
        action = np.asarray(action, dtype=np.float64).reshape(1, self.dim_act)
        if self.action_history_len == 1:
            return action.copy()
        return np.r_[self.action_history[1:], action]

    def _get_physics_features(self, obs_traj: np.ndarray) -> np.ndarray:
        if not self.include_physics_features or self.physics_feature_mode == "none":
            return np.empty(0, dtype=np.float64)
        if len(obs_traj) < 2:
            return np.zeros(2, dtype=np.float64)
        return ((obs_traj[-1] - obs_traj[-2]) * float(self.obs.freq)).astype(np.float64)

    def _get_chunk_state(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, Dict[str, float]]:
        ctrl_traj = np.asarray(action, dtype=np.float64).reshape(
            self.action_chunk_steps, self.dim_act
        )
        total_reward = 0.0
        reward_info: Dict[str, float] = {}
        state = None
        state_info: Dict[str, Any] = {}
        rewards = []
        for index, ctrl in enumerate(ctrl_traj):
            state, state_info = self.get_state(ctrl)
            step_reward, step_info = self.get_reward(state=state, action=ctrl)
            total_reward += (self.action_chunk_gamma ** index) * float(step_reward)
            rewards.append(float(step_reward))
            for key, value in step_info.items():
                reward_info[key] = float(value)
        reward_info["chunk_reward"] = float(total_reward)
        reward_info["chunk_step_0"] = rewards[0]
        reward_info["chunk_step_1"] = rewards[1]
        state_info["chunk_rewards"] = np.asarray(rewards, dtype=np.float64)
        return state, state_info, float(total_reward), reward_info

    def get_state(
        self,
        ctrl: np.ndarray
    ) -> np.ndarray:
        ctrl_traj = np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_act)
        if len(ctrl_traj) != 1:
            raise ValueError("get_state expects one two-valve command")
    
        # Reshape [[pos, neg], [pos, neg], ... ]
        goal_traj = self.goal.get_ref(self.t).reshape(-1,self.dim_obs) # next reference traj of time t

        # Get next observation o_{t+1}
        obs, obs_info = self.obs.observe(ctrl_traj.copy()[0], goal_traj.copy()[self.num_obs - 1])
        obs_traj = np.r_[
            self.obs_traj[1:],
            obs[1:3].reshape(-1,self.dim_obs)
        ]
        next_action_history = self._append_action_history(ctrl_traj[0])

        if self.pred is not None:
            pred_traj, pred_info = self.predict_obs(
                init_press = obs_traj[-1], 
                ctrl_traj = ctrl_traj,
                ref_traj = goal_traj[self.num_obs:self.num_obs + self.num_act]
            )
            state = np.r_[
                obs_traj.reshape(-1),
                next_action_history.reshape(-1),
                self._get_physics_features(obs_traj),
                pred_traj.reshape(-1), 
                goal_traj.reshape(-1)
            ]
        else:
            state = np.r_[
                obs_traj.reshape(-1),
                next_action_history.reshape(-1),
                self._get_physics_features(obs_traj),
                goal_traj.reshape(-1)
            ]

        # Update parameters
        self.t = obs[0] # t -> t+1
        self.obs_traj = obs_traj
        self.curr_obs = obs[1:3].astype(np.float32, copy=True)
        self.has_episode_state = True
        self.last_action = np.asarray(ctrl, dtype=np.float64).copy()
        self.action_history = next_action_history
        obs_info['ctrl'] = ctrl
        obs_info['pred'] = pred_info if self.pred is not None else None

        return state, obs_info

    def _state_from_current(
        self,
        ctrl: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build the next policy state without advancing the physical plant.

        A carried episode must not call ``obs.observe`` at reset time.  This
        helper rebuilds the reference/prediction part of the state while
        keeping the latest physical observation and simulator time intact.
        """
        ctrl_traj = np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_act)
        # ``PneuRef.buf`` already represents the current reference window
        # after the last physical step.  Reusing it avoids advancing the
        # reference twice at an artificial episode boundary.
        goal_traj = np.asarray(self.goal.buf, dtype=np.float64).copy().reshape(
            -1, self.dim_obs
        )
        obs_traj = np.asarray(self.obs_traj, dtype=np.float64).copy()
        current_obs = np.asarray(
            getattr(self.obs, "obs", self.curr_obs), dtype=np.float64
        )
        obs_traj[-1] = current_obs[0:2]
        current_physics = self._get_physics_features(obs_traj)

        pred_info = None
        if self.pred is not None:
            pred_traj, pred_info = self.predict_obs(
                init_press=obs_traj[-1],
                ctrl_traj=ctrl_traj,
                ref_traj=goal_traj[self.num_obs:self.num_obs + self.num_act],
            )
            state = np.r_[
                obs_traj.reshape(-1),
                self.action_history.reshape(-1),
                current_physics,
                pred_traj.reshape(-1),
                goal_traj.reshape(-1),
            ]
        else:
            state = np.r_[
                obs_traj.reshape(-1),
                self.action_history.reshape(-1),
                current_physics,
                goal_traj.reshape(-1),
            ]

        self.obs_traj = obs_traj
        self.curr_obs = obs_traj[-1].astype(np.float32, copy=True)
        state_info = dict(
            Observation=dict(
                curr_time=float(self.t),
                sen_pos=float(self.curr_obs[0]),
                sen_neg=float(self.curr_obs[1]),
                ref_pos=float(goal_traj[self.num_obs - 1, 0]),
                ref_neg=float(goal_traj[self.num_obs - 1, 1]),
                ctrl_pos=float(ctrl_traj[0, 0]),
                ctrl_neg=float(ctrl_traj[0, 1]),
            ),
            **{
                "obs_w/o_noise": np.r_[self.t, self.curr_obs].copy(),
                "ctrl": np.asarray(ctrl, dtype=np.float64).copy(),
                "pred": pred_info,
            },
        )
        return state, state_info
    
    def predict_obs(
        self,
        init_press: np.ndarray,
        ctrl_traj: np.ndarray,
        ref_traj: np.ndarray
    ) -> np.ndarray:
        self.pred.set_init_press(
            init_press[0],
            init_press[1]
        )
        rnd_num = 2*np.random.rand(2) - 1
        self.pred.set_offset(*(self.offset_range*rnd_num))

        preds = np.array([])
        for ctrl, goal in zip(ctrl_traj, ref_traj):
            pred, _ = self.pred.observe(ctrl, goal)
            preds = np.r_[preds, pred[1:3]]
        
        pred_info = dict(
            pred_act = ctrl_traj,
            pred_ref = ref_traj,
            pred_press = preds.reshape(-1,self.dim_obs)
        )
        
        return preds, pred_info
    
    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, dict[str, float]]:
        
        state = state.reshape(-1, self.dim_obs)

        obses = state[0:self.num_obs]
        # State layout is assembled in get_state() as:
        #
        #   observation history
        #   action history (optional)
        #   physics features (optional)
        #   predicted observations (optional)
        #   reference history/future
        #
        # Keep reward parsing synchronized with that layout.  Previously the
        # action-history and pressure-rate rows were skipped here, so the
        # reward compared pressure against valve commands/rates.  With rate
        # features this produced apparent errors of several thousand kPa.
        extra_feature_dims = self.dim_action_history + self.dim_physics_features
        if extra_feature_dims % self.dim_obs != 0:
            raise RuntimeError(
                "lib2 auxiliary state dimensions must align with pressure pairs: "
                f"{extra_feature_dims} is not divisible by {self.dim_obs}"
            )
        aux_rows = extra_feature_dims // self.dim_obs
        ref_base = self.num_obs + aux_rows
        if self.pred is not None:
            # Layout after auxiliary features:
            # predicted observations, reference history/future.
            pred_obses = state[ref_base:ref_base + self.num_act]
            ref_start = ref_base + self.num_act
            refs = state[ref_start:ref_start + self.num_obs]
            pred_refs = state[ref_start + self.num_obs:]
        else:
            ref_start = ref_base
            refs = state[ref_start:ref_start + self.num_obs]

        errs = refs - obses
        if self.pred is not None:
            pred_errs = pred_refs[0:self.num_act] - pred_obses
        prev_errs = errs[0:-1]
        curr_err = errs[-1]
        if self.pred is not None:
            pred_err = pred_errs[-1]
        # pred_errs = errs[self.num_prev + 1::]

        reward = 0

        pos_prev_reward = np.sum(np.abs(prev_errs[:,0]))
        pos_prev_reward *= - self.rwd_kwargs['pos_prev_rwd_coeff']
        neg_prev_reward = np.sum(np.abs(prev_errs[:,1]))
        neg_prev_reward *= - self.rwd_kwargs['neg_prev_rwd_coeff']
        reward += pos_prev_reward + neg_prev_reward

        pos_curr_reward = np.abs(curr_err[0])
        # pos_curr_reward = np.square(curr_err[0])
        pos_curr_reward *= - self.rwd_kwargs['pos_curr_rwd_coeff']
        neg_curr_reward = np.abs(curr_err[1])
        # neg_curr_reward = np.square(curr_err[1])
        neg_curr_reward *= - self.rwd_kwargs['neg_curr_rwd_coeff']
        reward += pos_curr_reward + neg_curr_reward
        
        if self.pred is not None:
            pos_fut_reward = np.sum(np.abs(pred_errs[:,0]))
            # pos_fut_reward = np.sum(np.square(pred_errs[:,0]))
            pos_fut_reward *= - self.rwd_kwargs['pos_fut_rwd_coeff']
            neg_fut_reward = np.sum(np.abs(pred_errs[:,1]))
            # neg_fut_reward = np.sum(np.square(pred_errs[:,1]))
            neg_fut_reward *= - self.rwd_kwargs['neg_fut_rwd_coeff']
            reward += pos_fut_reward + neg_fut_reward

            pos_pred_reward = np.abs(pred_err[0])
            # pos_pred_reward = np.square(pred_err[0])
            pos_pred_reward *= - self.rwd_kwargs['pos_pred_rwd_coeff']
            neg_pred_reward = np.abs(pred_err[1])
            # neg_pred_reward = np.square(pred_err[1])
            neg_pred_reward *= - self.rwd_kwargs['neg_pred_rwd_coeff']
            reward += pos_pred_reward + neg_pred_reward

        if self.pred is None:
            pos_fut_reward = 0
            neg_fut_reward = 0
            pos_pred_reward = 0
            neg_pred_reward = 0


        info = {
            'pos_prev_reward': pos_prev_reward,
            'neg_prev_reward': neg_prev_reward,
            'pos_curr_reward': pos_curr_reward,
            'neg_curr_reward': neg_curr_reward,
            'pos_fut_reward': pos_fut_reward,
            'neg_fut_reward': neg_fut_reward,
            'pos_pred_reward': pos_pred_reward,
            'neg_pred_reward': neg_pred_reward,
        }

        return reward, info
    
    def get_info(
        self,
        state_info: Dict[str, np.ndarray],
        reward_info: Dict[str, float]
    ) -> Dict[str, Union[np.ndarray, float]]:
        return dict(
            obs = state_info['Observation'],
            obs_wo_noise = state_info['obs_w/o_noise'],
            ctrl_input = state_info['ctrl'],
            pred = state_info['pred'],
            reward = reward_info
        )
    
    def verbose(
        self,
        info
    ):
        # np.set_printoptions(precision=4,linewidth=80, suppress=True)
        print(
            f'[ INFO] Pneumatic Env ==> \n'
            f'\tTime: {info["obs"]["curr_time"]}\n'
            f'\tSen : (\t{info["obs"]["sen_pos"]:3.4f}\t{info["obs"]["sen_neg"]:3.4f})\n'
            f'\tRef : (\t{info["obs"]["ref_pos"]:3.4f}\t{info["obs"]["ref_neg"]:3.4f})\n'
            f'\tCtrl: (\t{info["obs"]["ctrl_pos"]:3.4f}\t{info["obs"]["ctrl_neg"]:3.4f})\n'
            f'\tC/I : (\t{info["ctrl_input"][0]:3.4f}\t{info["ctrl_input"][1]:3.4f}) \n'
            f'\tw/o : {info["obs_wo_noise"]}\n'
            f'\tRWD : Curr \t{info["reward"]["pos_curr_reward"]:.4f}\t{info["reward"]["neg_curr_reward"]:.4f}\n'
            f'\t    : Prev \t{info["reward"]["pos_prev_reward"]:.4f}\t{info["reward"]["neg_prev_reward"]:.4f}\n'
            f'\t    : Fut  \t{info["reward"]["pos_fut_reward"]:.4f}\t{info["reward"]["neg_fut_reward"]:.4f}\n'
            f'\t    : Pred \t{info["reward"]["pos_pred_reward"]:.4f}\t{info["reward"]["neg_pred_reward"]:.4f}\n'
        )
        # )
        if self.verbose_clear:
            for _ in range(12):
                sys.stdout.write('\x1b[1A')  # move the cursor up one line
                sys.stdout.write('\x1b[2K')  # clear the line
        # for _ in range(8):
        #     sys.stdout.write("\033[f")  # move the cursor up one line
        #     sys.stdout.write("\033[k")  # clear the line
    
    def set_volume(self, vol1, vol2):
        self.obs.set_volume(vol1, vol2)
        self.pred.set_volume(vol1, vol2)

    def set_pid(
        self,
        Kp_pos: float,
        Ki_pos: float,
        Kd_pos: float,
        Kp_neg: float,
        Ki_neg: float,
        Kd_neg: float,
        Ka: Optional[float] = None
    ) -> None:
        self.obs.set_pid(
            Kp_pos, Ki_pos, Kd_pos,
            Kp_neg, Ki_neg, Kd_neg
        )
        if Ka is not None:
            self.obs.set_anti_windup(Ka)

if __name__ == '__main__':
    from pneu_ref.random_ref import RandomRef
    from sim import PneuSim
    from pred import PneuPred
    obs = PneuSim()
    ref = RandomRef()
    pred = PneuPred()
    env = PneuEnv(
        obs = obs,
        ref = ref,
        pred = pred
    )

    action = np.ones(env.action_space.shape, dtype=np.float64)
    state = env.reset()[0]
    print(env.step(action))
