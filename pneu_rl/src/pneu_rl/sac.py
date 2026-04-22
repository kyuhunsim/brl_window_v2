import os
import shutil
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
        noise_std: float = 0.1
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
    ):
        retrain_model_name = self.logger.set_retrain_model(
            is_model_loaded = self.is_model_loaded,
            retrain_model_name = retrain_model_name
        )
        
        self.load(
            name = retrain_model_name,
            train = True
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
            while epi_steps < self.horizon:
                if self.start_epi > self.last_epi + epi:
                    action = self.env.action_space.sample()
                else:
                    action = self.predict(state)
                
                next_state, reward, done, _, info = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                if len(self.buffer) > self.batch_size:
                    for _ in range(self.epoch):
                        self.update_parameters()
                
                if done:
                    break
                
                epi_reward += reward
                epi_steps += 1
                self.total_steps += 1

            epi += 1
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
                    temporal_weight = self.temporal_weight
                )
    
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
        state_dict = torch.load(path)
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
    ) -> None:
        self.set_logger(name)
        last_epi, last_steps = self.logger.load_infos(name)
        
        self.last_epi = last_epi
        self.total_steps = last_steps
        self.load_model(
            path = self.logger.model_path,
            evaluate = not train
        )

        self.buffer.load_buffer(
            path = self.logger.buffer_path
        )

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