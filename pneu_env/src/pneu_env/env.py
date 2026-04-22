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
            pos_diff_rwd_coeff = 0.0,
            neg_diff_rwd_coeff = 0.0,
        ),
        pos_pred_rnd_offset_range: float = 0,
        neg_pred_rnd_offset_range: float = 0
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

        self.dim_obs_traj = self.num_obs*self.dim_obs
        self.dim_fut_traj = self.num_act*self.dim_obs if pred is not None else 0
        self.dim_ref_traj = self.num_ref*self.dim_obs
        self.dim_act_traj = self.num_act*self.dim_act
        self.dim_state = self.dim_obs_traj + self.dim_fut_traj + self.dim_ref_traj
        
        # Env dimension
        self.observation_space = Box(
            low = -np.inf,
            high = np.inf,
            shape = (self.dim_state, ),
            dtype = np.float64
        )
        self.action_space = Box(
            low = -1.,
            high = 1.,
            shape = (self.dim_act_traj,),
            dtype = np.float64
        )

        # Initialize parameters
        self.publish_obs = False
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

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl = np.ones(self.action_space.shape[0], dtype=np.float64)
        state, _, _, _, info = self.step(ctrl)
        self.t = info['obs']['curr_time']

        if self.is_pid:
            self.obs.reset_pid()
        
        return state, info 
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        state, state_info = self.get_state(action)
        reward, reward_info = self.get_reward(
            state = state,
            action = action
        )
        terminated = False
        truncated = False

        info = self.get_info(state_info, reward_info) 
        
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

    def get_state(
        self,
        ctrl: np.ndarray
    ) -> np.ndarray:
        ctrl_traj = ctrl.reshape(-1, self.dim_act)
    
        # Reshape [[pos, neg], [pos, neg], ... ]
        goal_traj = self.goal.get_ref(self.t).reshape(-1,self.dim_obs) # next reference traj of time t

        # Get next observation o_{t+1}
        obs, obs_info = self.obs.observe(ctrl_traj.copy()[0], goal_traj.copy()[self.num_obs - 1])
        obs_traj = np.r_[
            self.obs_traj[1:],
            obs[1:3].reshape(-1,self.dim_obs)
        ]

        if self.pred is not None:
            pred_traj, pred_info = self.predict_obs(
                init_press = obs_traj[-1], 
                ctrl_traj = ctrl_traj,
                ref_traj = goal_traj[self.num_obs:self.num_obs + self.num_act]
            )
            state = np.r_[
                obs_traj.reshape(-1), 
                pred_traj.reshape(-1), 
                goal_traj.reshape(-1)
            ]
        else:
            state = np.r_[
                obs_traj.reshape(-1), 
                goal_traj.reshape(-1)
            ]

        # Update parameters
        self.t = obs[0] # t -> t+1
        self.obs_traj = obs_traj
        obs_info['ctrl'] = ctrl
        obs_info['pred'] = pred_info if self.pred is not None else None

        return state, obs_info
    
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
        if self.pred is not None:
            pred_obses = state[self.num_obs:self.num_obs + self.num_act]
        refs = state[self.num_obs + self.num_act:2*self.num_obs + self.num_act]
        pred_refs = state[2*self.num_obs + self.num_act::]

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

        # diff_errs = errs[1:] - errs[:-1]
        # pos_diff_reward = np.sum(np.abs(diff_errs[:,0]))
        # pos_diff_reward *= - self.rwd_kwargs['pos_diff_rwd_coeff']
        # neg_diff_reward = np.sum(np.abs(diff_errs[:,1]))
        # neg_diff_reward *= - self.rwd_kwargs['neg_diff_rwd_coeff']
        # reward += pos_diff_reward + neg_diff_reward

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
            # 'pos_diff_reward': pos_diff_reward,
            # 'neg_diff_reward': neg_diff_reward,
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
            # f'\tPRED:\n'
            # f'   ACT              PRESS             REF\n'
            # f'{np.hstack((info["pred"]["pred_act"].reshape(-1,2), info["pred"]["pred_press"].reshape(-1,2), info["pred"]["pred_ref"].reshape(-1,2)))}'
            # f'\t      Act\t{info["reward"]["pos_act_reward"]}\t{info["reward"]["neg_act_reward"]}\n'
            # f'\t      Diff\t{info["reward"]["pos_diff_reward"]}\t{info["reward"]["neg_diff_reward"]}'
        # )
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
