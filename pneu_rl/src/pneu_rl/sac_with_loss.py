import os
import shutil
import csv
from typing import Any, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import math

import gymnasium as gym

from pneu_rl.networks import SACPolicy, QNetwork
from pneu_rl.buffer import ReplayBuffer
from pneu_rl.logger import Logger
from pneu_env.env import PneuEnv
from pneu_utils.utils import get_pkg_path, color

class SAC():
    def __init__(
        self,
        env: Union[gym.Env, PneuEnv],
        learning_rate: float = 1e-4,
        gamma: float = 0.8,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tunning: bool = True,
        hidden_dim: int = 64,
        buffer_size: int = 25e4,
        batch_size: int = 128,
        epoch: int = 1,
        horizon: int = 2048,
        start_epi: int = 10,
        max_grad_norm: float = 0.5,
        log_std_min: float = -10,
        log_std_max: float = 1,
        temporal_weight: float = 0.1,
        spatial_weight: float = 0.5,
        noise_std: float = 0.1,
        train_diag_interval: int = 1,
        train_diag_atm_band: float = 8.0,
        update_every: int = 1,
    ):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.policy = SACPolicy(
            state_dim = state_dim, 
            action_dim = action_dim, 
            hidden_dim = hidden_dim,
            log_std_min = log_std_min,
            log_std_max = log_std_max,
            action_space = env.action_space
        )
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        self.critic = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.hard_update(self.critic_target, self.critic)

        self.auto_ent = automatic_entropy_tunning
        if self.auto_ent:
            self.target_ent = -torch.prod(torch.tensor((action_dim,))).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha
            self.log_alpha = math.log(alpha)

        self.buffer = ReplayBuffer(buffer_size, noise_std, env.dim_ref_traj)

        self.gamma = gamma
        self.tau = tau
        
        self.temporal_weight = temporal_weight
        self.temporal_weight_hardening = False
        self.spatial_weight = spatial_weight

        self.max_grad_norm = max_grad_norm

        self.horizon = horizon
        self.start_epi = start_epi
        self.batch_size = batch_size
        self.epoch = epoch

        self.last_epi = 0
        self.total_steps = 0
        self.train_diag_interval = max(int(train_diag_interval), 0)
        self.train_diag_atm_band = float(train_diag_atm_band)
        self.update_every = max(int(update_every), 1)

        self.log = False
        self.model_name = 'SAC'
        self.is_model_loaded = False
    
    def set_logger(
        self,
        model_name: str
    ) -> None:
        self.model_name = model_name
        self.log = True
        self.logger = Logger(model_name) 

    def set_temporal_weight_hardening(
        self,
        initial_weight: float,
        max_weight: float,
        rate: float
    ) -> None:
        self.temporal_weight_hardening = True
        self.temporal_weight = initial_weight
        self.max_temporal_weight = max_weight
        self.increase_rate = rate
    
    def set_retrain(
        self,
        retrain_model_name: Optional[str] = None,
        load_buffer: bool = True,
        copy_buffer: bool = True,
    ):
        retrain_model_name = self.logger.set_retrain_model(
            is_model_loaded = self.is_model_loaded,
            retrain_model_name = retrain_model_name,
            copy_buffer = copy_buffer,
        )
        
        self.load(
            name = retrain_model_name,
            train = True,
            load_buffer = load_buffer,
        )
        
        print(f'[ INFO] Retrain Model Name: {retrain_model_name}')

        return retrain_model_name

    def set_alpha(
        self,
        alpha: float,
        automatic_entropy_tunning: bool = True
    ):
        self.auto_ent = automatic_entropy_tunning
        if self.auto_ent:
            self.target_ent = -torch.prod(torch.tensor((self.action_dim,))).item()
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.alpha = torch.tensor(alpha, requires_grad=False)
            self.log_alpha = torch.log(self.alpha)
    
    def clear_buffer(self) -> None:
        self.buffer.clear_buffer()
    
    def predict(
        self,
        state: np.ndarray,
        evaluate: bool = True
    ):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        
        return action.detach().numpy()[0]

    def select_action(
        self,
        state: np.ndarray,
    ):
        return self.predict(state, evaluate=False)

    def evaluate_action(
        self,
        state: np.ndarray,
    ):
        return self.predict(state, evaluate=True)
    
    def update_parameters(
        self,
    ) -> None:
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
            noise_state_batch
        ) = self.buffer.sample(batch_size = self.batch_size)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha*next_log_prob
            # min_qf_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch*self.gamma*min_qf_target
        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.smooth_l1_loss(qf1, next_q_value)
        qf2_loss = F.smooth_l1_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.current_critics_loss = qf_loss.item()

        self.critic_optim.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm
        )
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        next_pi, _, _ = self.policy.sample(next_state_batch)
        noise_pi, _, _ = self.policy.sample(noise_state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        pi_loss = ((self.alpha*log_pi) - min_qf_pi).mean()
        
        L_temporal = self.temporal_weight*torch.norm(pi - next_pi, p=2) # check sign!!
        L_spatial = self.spatial_weight*torch.norm(pi - noise_pi, p=2) # check sign!!
        policy_loss = pi_loss + L_temporal + L_spatial
        # policy_loss = pi_loss

        self.current_policy_loss = policy_loss.item()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm
        )
        self.policy_optim.step()

        if self.auto_ent:
            alpha_loss = -(self.log_alpha*(log_pi + self.target_ent).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        
        if self.temporal_weight_hardening:
            self.harden_temporal_weight()

        self.soft_update(self.critic_target, self.critic, self.tau)

    def train(
        self,
        episode: int
    ) -> None:
        epi = 0
        while epi < episode:
            epi_reward = 0
            done = False

            state = self.env.reset()[0]

            epi_steps = 0
            total_critic_loss = 0
            total_policy_loss = 0
            update_count = 0
            episode_infos = []

            while epi_steps < self.horizon:
                if self.start_epi > self.last_epi + epi:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)
                
                next_state, reward, done, _, info = self.env.step(action)
                episode_infos.append(info)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                if len(self.buffer) > self.batch_size and self.total_steps % self.update_every == 0:
                    for _ in range(self.epoch):
                        self.update_parameters()
                        total_critic_loss += self.current_critics_loss
                        total_policy_loss += self.current_policy_loss
                        update_count += 1
                
                if done:
                    break
                
                epi_reward += reward
                epi_steps += 1
                self.total_steps += 1

            epi += 1
            avg_critic_loss = total_critic_loss/update_count if update_count > 0 else 0
            avg_policy_loss = total_policy_loss/update_count if update_count > 0 else 0

            print(
                f'[ INFO] {self.model_name} ==> |epi|\t{self.last_epi + epi}\t'
                f'|step|\t{self.total_steps}\t'
                f'|reward|\t{epi_reward:.4f}'
            )
            
            if self.log:
                self.save_model(self.logger.model_path)
                if epi % 100 == 0:
                    self.buffer.save_buffer(self.logger.buffer_path)
                self.logger.save_infos(
                    epi = self.last_epi + epi,
                    reward = epi_reward,
                    step = self.total_steps,
                    alpha = self.alpha,
                    temporal_weight = self.temporal_weight,
                    critic_loss = avg_critic_loss,
                    policy_loss = avg_policy_loss
                )
                if self.train_diag_interval > 0 and epi % self.train_diag_interval == 0:
                    diag = self._episode_diagnostics(
                        epi=self.last_epi + epi,
                        reward=epi_reward,
                        infos=episode_infos,
                    )
                    self._append_train_diagnostics(diag)

    def warmup_buffer(
        self,
        steps: int,
        deterministic: bool = False,
        reset_on_horizon: bool = True,
    ) -> None:
        steps = int(steps)
        if steps <= 0:
            return

        state = self.env.reset()[0]
        epi_steps = 0
        reward_sum = 0.0
        for step in range(steps):
            if deterministic:
                action = self.evaluate_action(state)
            else:
                action = self.select_action(state)

            next_state, reward, done, _, _ = self.env.step(action)
            self.buffer.add(state, action, reward, next_state, done)
            state = next_state
            reward_sum += float(reward)
            epi_steps += 1
            self.total_steps += 1

            if done or (reset_on_horizon and epi_steps >= self.horizon):
                state = self.env.reset()[0]
                epi_steps = 0

        print(
            f"[ INFO] Warmup buffer collected: steps={steps}, "
            f"buffer={len(self.buffer)}, reward_sum={reward_sum:.4f}"
        )

    def _episode_diagnostics(
        self,
        epi: int,
        reward: float,
        infos: list[dict],
    ) -> dict[str, float]:
        if len(infos) == 0:
            return dict(epi=epi, reward=reward, n=0)

        def arr(path: tuple[str, str]) -> np.ndarray:
            return np.asarray([info[path[0]][path[1]] for info in infos], dtype=np.float64)

        time_arr = arr(("obs", "curr_time"))
        ch_pos = arr(("obs", "sen_pos"))
        ch_neg = arr(("obs", "sen_neg"))
        act_pos = arr(("obs", "sen_act_pos"))
        act_neg = arr(("obs", "sen_act_neg"))
        ref_pos = arr(("obs", "ref_act_pos"))
        ref_neg = arr(("obs", "ref_act_neg"))
        u_ch_pos = arr(("obs", "ctrl_pos"))
        u_ch_neg = arr(("obs", "ctrl_neg"))
        u_pos_in = arr(("obs", "ctrl_act_pos_in"))
        u_pos_out = arr(("obs", "ctrl_act_pos_out"))
        u_neg_in = arr(("obs", "ctrl_act_neg_in"))
        u_neg_out = arr(("obs", "ctrl_act_neg_out"))

        pos_err = ref_pos - act_pos
        neg_err = ref_neg - act_neg
        err_deadband = 2.0
        pos_need_up = pos_err > err_deadband
        pos_need_down = pos_err < -err_deadband
        neg_need_up = neg_err > err_deadband
        neg_need_down = neg_err < -err_deadband

        def rmse(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0

        def mean_when(x: np.ndarray, mask: np.ndarray) -> float:
            return float(np.mean(x[mask])) if bool(np.any(mask)) else 0.0

        def ratio_when(cond: np.ndarray, mask: np.ndarray) -> float:
            return float(np.mean(cond[mask])) if bool(np.any(mask)) else 0.0

        def slope(y: np.ndarray) -> float:
            ok = np.isfinite(time_arr) & np.isfinite(y)
            if int(np.sum(ok)) < 3:
                return 0.0
            return float(np.polyfit(time_arr[ok], y[ok], 1)[0])

        def sign_change_per_sec(y: np.ndarray) -> float:
            if len(y) < 4:
                return 0.0
            dy = np.diff(y)
            sign = np.sign(dy)
            sign = sign[sign != 0]
            if len(sign) < 2:
                return 0.0
            duration = max(float(time_arr[-1] - time_arr[0]), 1e-9)
            return float(np.mean(sign[1:] != sign[:-1]) * len(sign) / duration)

        def delta_abs_mean(y: np.ndarray) -> float:
            return float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else 0.0

        atm = 101.325
        support_band = 8.0
        act_pos_near_atm = np.abs(act_pos - atm) < self.train_diag_atm_band
        act_neg_near_atm = np.abs(act_neg - atm) < self.train_diag_atm_band
        pos_support_margin = ch_pos - act_pos
        neg_relief_margin = act_neg - ch_neg
        neg_fill_margin = atm - act_neg

        return dict(
            epi=int(epi),
            reward=float(reward),
            n=int(len(infos)),
            duration_sec=float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0,
            pos_rmse=rmse(pos_err),
            neg_rmse=rmse(neg_err),
            pos_bias=float(np.mean(pos_err)),
            neg_bias=float(np.mean(neg_err)),
            ch_pos_start=float(ch_pos[0]),
            ch_pos_end=float(ch_pos[-1]),
            ch_pos_max=float(np.max(ch_pos)),
            ch_pos_slope=slope(ch_pos),
            ch_neg_start=float(ch_neg[0]),
            ch_neg_end=float(ch_neg[-1]),
            ch_neg_min=float(np.min(ch_neg)),
            ch_neg_slope=slope(ch_neg),
            pos_conflict_mean=float(np.mean(u_pos_in * u_pos_out)),
            neg_conflict_mean=float(np.mean(u_neg_in * u_neg_out)),
            pos_need_up_ratio=float(np.mean(pos_need_up)),
            pos_need_down_ratio=float(np.mean(pos_need_down)),
            pos_wrong_out_when_need_up=mean_when(u_pos_out, pos_need_up),
            pos_wrong_in_when_need_down=mean_when(u_pos_in, pos_need_down),
            pos_wrong_gt_right_when_need_up=ratio_when(u_pos_out > u_pos_in, pos_need_up),
            pos_wrong_gt_right_when_need_down=ratio_when(u_pos_in > u_pos_out, pos_need_down),
            pos_support_margin_when_need_up=mean_when(pos_support_margin, pos_need_up),
            pos_support_low_ratio_when_need_up=ratio_when(pos_support_margin < support_band, pos_need_up),
            neg_need_up_ratio=float(np.mean(neg_need_up)),
            neg_need_down_ratio=float(np.mean(neg_need_down)),
            neg_wrong_in_when_need_up=mean_when(u_neg_in, neg_need_up),
            neg_wrong_out_when_need_down=mean_when(u_neg_out, neg_need_down),
            neg_wrong_gt_right_when_need_up=ratio_when(u_neg_in > u_neg_out, neg_need_up),
            neg_wrong_gt_right_when_need_down=ratio_when(u_neg_out > u_neg_in, neg_need_down),
            neg_fill_margin_when_need_up=mean_when(neg_fill_margin, neg_need_up),
            neg_fill_low_ratio_when_need_up=ratio_when(neg_fill_margin < support_band, neg_need_up),
            neg_relief_margin_when_need_down=mean_when(neg_relief_margin, neg_need_down),
            neg_relief_low_ratio_when_need_down=ratio_when(neg_relief_margin < support_band, neg_need_down),
            act_pos_near_atm_ratio=float(np.mean(act_pos_near_atm)),
            act_neg_near_atm_ratio=float(np.mean(act_neg_near_atm)),
            act_pos_sign_change_per_sec=sign_change_per_sec(act_pos),
            act_neg_sign_change_per_sec=sign_change_per_sec(act_neg),
            ch_pos_sign_change_per_sec=sign_change_per_sec(ch_pos),
            ch_neg_sign_change_per_sec=sign_change_per_sec(ch_neg),
            u_ch_pos_mean=float(np.mean(u_ch_pos)),
            u_ch_neg_mean=float(np.mean(u_ch_neg)),
            u_pos_in_mean=float(np.mean(u_pos_in)),
            u_pos_out_mean=float(np.mean(u_pos_out)),
            u_neg_in_mean=float(np.mean(u_neg_in)),
            u_neg_out_mean=float(np.mean(u_neg_out)),
            u_ch_pos_delta=delta_abs_mean(u_ch_pos),
            u_ch_neg_delta=delta_abs_mean(u_ch_neg),
            u_pos_in_delta=delta_abs_mean(u_pos_in),
            u_pos_out_delta=delta_abs_mean(u_pos_out),
            u_neg_in_delta=delta_abs_mean(u_neg_in),
            u_neg_out_delta=delta_abs_mean(u_neg_out),
        )

    def _append_train_diagnostics(
        self,
        diag: dict[str, float],
    ) -> None:
        path = f"{self.logger.folder_path}/train_diagnostics.csv"
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(diag)
    
    def save_model(
        self,
        path: str = 'model.pth'
    ):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'log_alpha': self.log_alpha
        }, path)

    def load_model(
        self,
        path: str = 'model.pth',
        evaluate: bool = True
    ) -> None:
        state_dict = torch.load(path, weights_only=True)
        self.policy.load_state_dict(state_dict['policy_state_dict'])
        self.critic.load_state_dict(state_dict['critic_state_dict'])
        self.critic_target.load_state_dict(state_dict['critic_target_state_dict'])
        self.policy_optim.load_state_dict(state_dict['policy_optimizer_state_dict'])
        self.critic_optim.load_state_dict(state_dict['critic_optimizer_state_dict'])
        self.log_alpha = state_dict['log_alpha']
        
        if self.auto_ent:
            if not self.log_alpha.requires_grad:
                self.log_alpha.requires_grad_()  # requires_grad 보장
            self.alpha_optim = Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha = self.log_alpha.exp()


        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()
    
    def load(
        self,
        name: str,
        train: bool = False,
        load_buffer: bool = True,
    ) -> None:
        self.set_logger(name)
        last_epi, last_steps = self.logger.load_infos(name)
        
        self.last_epi = last_epi
        self.total_steps = last_steps
        self.load_model(
            path = self.logger.model_path,
            evaluate = not train
        )

        if load_buffer:
            self.buffer.load_buffer(
                path = self.logger.buffer_path
            )
        else:
            self.buffer.clear_buffer()

        self.is_model_loaded = True

    def soft_update(
        self,
        target: QNetwork,
        source: QNetwork,
        tau: float
    ) -> None:
        for target_param, param in zip(
            target.parameters(),
            source.parameters()
        ):
            target_param.data.copy_(
                target_param.data*(1.0 - tau) + param.data*tau
            )
        
    def hard_update(
        self,
        target: QNetwork,
        source: QNetwork
    ) -> None:
        for target_param, param in zip(
            target.parameters(), 
            source.parameters()
        ):
            target_param.data.copy_(param.data)

    def harden_temporal_weight(self) -> None:
        increased_temporal_weight = self.temporal_weight + self.increase_rate
        self.temporal_weight = min(max(self.temporal_weight, increased_temporal_weight), self.max_temporal_weight)
