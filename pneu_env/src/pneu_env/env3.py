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
        rwd_kwargs: Dict[str, float] = dict(
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_curr_rwd_coeff=0.0,
            neg_curr_rwd_coeff=0.0,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
            pos_diff_rwd_coeff=0.0,
            neg_diff_rwd_coeff=0.0,
        ),
        pos_pred_rnd_offset_range: float = 0,
        neg_pred_rnd_offset_range: float = 0,
        verbose: bool = True,
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

        self.num_obs = num_prev + 1
        self.num_ref = num_prev + num_pred + 1

        self.dim_obs = 4
        self.dim_ref = 2
        self.dim_act = 6

        self.dim_obs_traj = self.num_obs * self.dim_obs
        self.dim_fut_traj = self.num_act * self.dim_obs if pred is not None else 0
        self.dim_ref_traj = self.num_ref * self.dim_ref
        self.dim_act_traj = self.num_act * self.dim_act
        self.dim_state = self.dim_obs_traj + self.dim_fut_traj + self.dim_ref_traj

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dim_state,),
            dtype=np.float64,
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
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
        self.t = 0.0
        self.rwd_kwargs = rwd_kwargs
        self.offset_range = np.array(
            [pos_pred_rnd_offset_range, neg_pred_rnd_offset_range],
            dtype=np.float64,
        )
        self.publish_obs = False
        self.is_pid = False
        self.verbose_enabled = bool(verbose)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl = np.zeros(self.action_space.shape[0], dtype=np.float64)
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

        return state, reward, terminated, truncated, info

    def close(self) -> None:
        ctrl = np.zeros(self.dim_act, dtype=np.float64)
        goal = np.array([101.325, 101.325], dtype=np.float64)
        _, _ = self.obs.observe(ctrl, goal)

    def get_state(
        self,
        ctrl: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl_traj = np.asarray(ctrl, dtype=np.float64).reshape(-1, self.dim_act)
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
        obs_info["ctrl"] = ctrl
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
        for ctrl, goal in zip(ctrl_traj, ref_traj):
            pred, _ = self.pred.observe(ctrl, goal)
            preds = np.r_[preds, pred[1:5]]

        pred_press = preds.reshape(-1, self.dim_obs)
        pred_info = dict(
            pred_act=ctrl_traj,
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
        refs = ref_traj[:self.num_obs]
        errs = refs - act_obses

        prev_errs = errs[0:-1]
        curr_err = errs[-1]

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

        if self.pred is not None and pred_traj is not None:
            pred_act_obses = pred_traj[:, 2:4]
            pred_refs = ref_traj[self.num_obs:self.num_obs + self.num_act]
            pred_errs = pred_refs - pred_act_obses
            pred_err = pred_errs[-1]

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
        else:
            pos_fut_reward = 0.0
            neg_fut_reward = 0.0
            pos_pred_reward = 0.0
            neg_pred_reward = 0.0

        info = {
            "pos_prev_reward": pos_prev_reward,
            "neg_prev_reward": neg_prev_reward,
            "pos_curr_reward": pos_curr_reward,
            "neg_curr_reward": neg_curr_reward,
            "pos_fut_reward": pos_fut_reward,
            "neg_fut_reward": neg_fut_reward,
            "pos_pred_reward": pos_pred_reward,
            "neg_pred_reward": neg_pred_reward,
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
        print(
            f'[ INFO] Pneumatic Env3 ==> \n'
            f'\tTime: {info["obs"]["curr_time"]}\n'
            f'\tCh  : (\t{info["obs"]["sen_pos"]:3.4f}\t{info["obs"]["sen_neg"]:3.4f})\n'
            f'\tAct : (\t{info["obs"]["sen_act_pos"]:3.4f}\t{info["obs"]["sen_act_neg"]:3.4f})\n'
            f'\tRef : (\t{info["obs"]["ref_act_pos"]:3.4f}\t{info["obs"]["ref_act_neg"]:3.4f})\n'
            f'\tCtrl: {np.array2string(np.asarray(info["ctrl_input"])[0:self.dim_act], precision=4)}\n'
            f'\tw/o : {info["obs_wo_noise"]}\n'
            f'\tRWD : Curr \t{info["reward"]["pos_curr_reward"]:.4f}\t{info["reward"]["neg_curr_reward"]:.4f}\n'
            f'\t    : Prev \t{info["reward"]["pos_prev_reward"]:.4f}\t{info["reward"]["neg_prev_reward"]:.4f}\n'
            f'\t    : Fut  \t{info["reward"]["pos_fut_reward"]:.4f}\t{info["reward"]["neg_fut_reward"]:.4f}\n'
            f'\t    : Pred \t{info["reward"]["pos_pred_reward"]:.4f}\t{info["reward"]["neg_pred_reward"]:.4f}\n'
        )
        for _ in range(13):
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
