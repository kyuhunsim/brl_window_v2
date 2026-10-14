from __future__ import annotations

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
from pneu_rl.exploration import PinkNoise
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
        target_entropy: float | None = None,
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
        caps_mode: str = "legacy",
        caps_spatial_noise_normalized: float = 0.01,
        caps_start_steps: int = 0,
        caps_ramp_steps: int = 0,
        normalize_observation: bool = False,
        train_diag_interval: int = 1,
        train_diag_atm_band: float = 8.0,
        train_verbose_interval: int = 0,
        exploration_noise: str = "none",
        exploration_noise_std: float = 0.0,
        exploration_noise_octaves: int = 8,
        exploration_noise_seed: int = 0,
        exploration_use_policy_mean: bool = False,
        warmup_action_raw: float | np.ndarray | list[float] | None = None,
        warmup_uniform_low: float | np.ndarray | list[float] | None = None,
        warmup_mode: str = "fixed",
        warmup_noise_std: float = 0.0,
        warmup_noise_octaves: int = 8,
        warmup_noise_seed: int = 0,
        update_during_warmup: bool = True,
        reset_after_warmup: bool = False,
        # Kept only so older cfg.yaml files can still be loaded.  These
        # legacy action-wrapper settings are intentionally ignored.
        warmup_action: float | np.ndarray | list[float] = 0.5,
        policy_init_action: float | np.ndarray | list[float] | None = None,
        warmup_action_schedule: list[list[float]] | None = None,
        feasible_action_points: list[list[float]] | None = None,
        action_hold_steps: int = 1,
        command_transition_max_delta: float = 0.0,
        update_every: int = 1,
        chunk_consistency_weight: float = 0.0,
        checkpoint_interval: int = 0,
    ):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.normalize_observation = bool(normalize_observation)
        state_scale = np.ones(state_dim, dtype=np.float32)
        if self.normalize_observation:
            if not hasattr(env, "policy_state_scale"):
                raise ValueError(
                    "normalize_observation=True requires env.policy_state_scale"
                )
            state_scale = np.asarray(env.policy_state_scale, dtype=np.float32)
            if state_scale.shape != (state_dim,):
                raise ValueError(
                    "policy_state_scale must match observation dimension: "
                    f"{state_scale.shape} != {(state_dim,)}"
                )
            if np.any(~np.isfinite(state_scale)) or np.any(state_scale <= 0.0):
                raise ValueError("policy_state_scale must be finite and positive")
        self.state_scale = torch.as_tensor(state_scale, dtype=torch.float32)

        self.policy = SACPolicy(
            state_dim = state_dim, 
            action_dim = action_dim, 
            hidden_dim = hidden_dim,
            log_std_min = log_std_min,
            log_std_max = log_std_max,
            action_space = env.action_space,
            initial_action=policy_init_action,
        )
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        self.critic = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.hard_update(self.critic_target, self.critic)

        self.auto_ent = automatic_entropy_tunning
        if self.auto_ent:
            self.target_ent = (
                float(target_entropy)
                if target_entropy is not None
                else -torch.prod(torch.tensor((action_dim,))).item()
            )
            self.log_alpha = torch.tensor(
                [math.log(max(float(alpha), 1e-8))],
                dtype=torch.float32,
                requires_grad=True,
            )
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha
            self.log_alpha = math.log(alpha)

        self.buffer = ReplayBuffer(buffer_size, noise_std, env.dim_ref_traj)

        self.gamma = gamma
        # A chunked environment step represents K sequential simulator
        # transitions.  The environment already returns the discounted sum
        # of the K rewards, so the bootstrap must advance by gamma**K.
        self.action_chunk_steps = max(
            int(getattr(env, "action_chunk_steps", 1)),
            1,
        )
        self.action_chunk_mode = str(getattr(env, "action_chunk_mode", "free"))
        self.chunk_consistency_weight = max(float(chunk_consistency_weight), 0.0)
        self.tau = tau
        
        self.temporal_weight = temporal_weight
        self.temporal_weight_hardening = False
        self.spatial_weight = spatial_weight
        self.caps_mode = str(caps_mode).lower()
        if self.caps_mode not in ("legacy", "mean_action"):
            raise ValueError(
                "caps_mode must be 'legacy' or 'mean_action', "
                f"got {self.caps_mode!r}"
            )
        self.caps_spatial_noise_normalized = max(
            float(caps_spatial_noise_normalized), 0.0
        )
        self.caps_start_steps = max(int(caps_start_steps), 0)
        self.caps_ramp_steps = max(int(caps_ramp_steps), 0)
        self.current_caps_factor = 0.0

        self.max_grad_norm = max_grad_norm

        self.horizon = horizon
        self.start_epi = start_epi
        action_low = np.asarray(self.env.action_space.low, dtype=np.float64)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float64)
        self.warmup_uniform_low = action_low.copy()
        if warmup_uniform_low is not None:
            uniform_low = np.asarray(warmup_uniform_low, dtype=np.float64).reshape(-1)
            if uniform_low.size == 1:
                uniform_low = np.full(action_dim, float(uniform_low[0]), dtype=np.float64)
            if uniform_low.size != action_dim:
                raise ValueError(
                    "warmup_uniform_low must be scalar or match action dimension: "
                    f"got {uniform_low.size} for action_dim={action_dim}"
                )
            if np.any(uniform_low < action_low) or np.any(uniform_low > action_high):
                raise ValueError(
                    "warmup_uniform_low must lie inside env.action_space: "
                    f"got {uniform_low} for [{action_low}, {action_high}]"
                )
            self.warmup_uniform_low = uniform_low
        self.warmup_action_raw = None
        if warmup_action_raw is not None:
            warmup = np.asarray(warmup_action_raw, dtype=np.float64).reshape(-1)
            if warmup.size == 1:
                warmup = np.full(action_dim, float(warmup[0]), dtype=np.float64)
            if warmup.size != action_dim:
                raise ValueError(
                    "warmup_action_raw must be scalar or match action dimension: "
                    f"got {warmup.size} for action_dim={action_dim}"
                )
            if np.any(warmup < action_low) or np.any(warmup > action_high):
                raise ValueError(
                    "warmup_action_raw must lie inside env.action_space: "
                    f"got {warmup} for [{action_low}, {action_high}]"
                )
            self.warmup_action_raw = warmup
        self.warmup_mode = str(warmup_mode).lower()
        if self.warmup_mode not in ("fixed", "uniform", "pink"):
            raise ValueError(
                "warmup_mode must be 'fixed', 'uniform', or 'pink', "
                f"got {self.warmup_mode!r}"
            )
        self.warmup_noise_std = max(float(warmup_noise_std), 0.0)
        self.warmup_noise = (
            PinkNoise(
                dim=action_dim,
                seed=int(warmup_noise_seed),
                octaves=warmup_noise_octaves,
            )
            if self.warmup_mode == "pink" and self.warmup_noise_std > 0.0
            else None
        )
        self.update_during_warmup = bool(update_during_warmup)
        self.reset_after_warmup = bool(reset_after_warmup)
        self.batch_size = batch_size
        self.epoch = epoch

        self.last_epi = 0
        self.total_steps = 0
        self.checkpoint_interval = max(int(checkpoint_interval), 0)
        self.train_diag_interval = max(int(train_diag_interval), 0)
        self.train_diag_atm_band = float(train_diag_atm_band)
        self.train_verbose_interval = max(int(train_verbose_interval), 0)
        self.exploration_noise = str(exploration_noise).lower()
        self.exploration_noise_std = max(float(exploration_noise_std), 0.0)
        self.exploration_use_policy_mean = bool(exploration_use_policy_mean)
        if self.exploration_noise not in ("none", "pink"):
            raise ValueError(
                "exploration_noise must be 'none' or 'pink', "
                f"got {self.exploration_noise!r}"
            )
        self.pink_noise = (
            PinkNoise(
                dim=action_dim,
                seed=int(exploration_noise_seed),
                octaves=exploration_noise_octaves,
            )
            if self.exploration_noise == "pink"
            else None
        )
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

    def _normalize_state_tensor(self, state: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observation:
            return state
        return state / self.state_scale.to(device=state.device, dtype=state.dtype)

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self._normalize_state_tensor(tensor)

    def _policy_mean_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        raw_mean, _ = self.policy.forward(state)
        return (
            self.policy.action_scale * torch.tanh(raw_mean)
            + self.policy.action_bias
        )

    def _caps_factor(self) -> float:
        if self.total_steps < self.caps_start_steps:
            return 0.0
        if self.caps_ramp_steps <= 0:
            return 1.0
        progress = (self.total_steps - self.caps_start_steps) / float(
            self.caps_ramp_steps
        )
        return float(np.clip(progress, 0.0, 1.0))
    
    def predict(
        self,
        state: np.ndarray,
        evaluate: bool = True
    ):
        state = self._state_tensor(state)
        
        if evaluate:
            with torch.no_grad():
                _, _, action = self.policy.sample(state)
            return action.detach().numpy()[0]
        else:
            action, _, _ = self.policy.sample(state)

        return action.detach().numpy()[0]

    def reset_action_state(self) -> None:
        """Compatibility no-op for evaluation scripts from older runs."""
        return None

    def select_action(
        self,
        state: np.ndarray,
    ):
        action, _ = self.select_action_with_diagnostics(state)
        return action

    def select_action_with_diagnostics(
        self,
        state: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        """Sample an action and return policy statistics for exploration logs."""
        state_tensor = self._state_tensor(state)
        with torch.no_grad():
            action, log_prob, policy_mean = self.policy.sample(state_tensor)
            _, log_std = self.policy.forward(state_tensor)
        policy_action = (
            policy_mean.numpy()[0]
            if self.exploration_use_policy_mean
            else action.numpy()[0]
        )
        pink = self._sample_exploration_noise()
        sampled_action = self._apply_exploration_noise(policy_action, pink)
        action_np = sampled_action.copy()
        return action_np, dict(
            log_prob=float(log_prob.numpy()[0, 0]),
            log_std=log_std.numpy()[0].copy(),
            action=action_np.copy(),
            policy_action=policy_action.copy(),
            pink_noise=pink.copy(),
        )

    def _sample_exploration_noise(self) -> np.ndarray:
        if self.pink_noise is None or self.exploration_noise_std <= 0.0:
            return np.zeros(self.action_dim, dtype=np.float64)
        return self.pink_noise.sample()

    def _apply_exploration_noise(
        self,
        action: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if self.pink_noise is not None and self.exploration_noise_std > 0.0:
            action = action + self.exploration_noise_std * noise
        low = np.asarray(self.env.action_space.low, dtype=np.float64)
        high = np.asarray(self.env.action_space.high, dtype=np.float64)
        return np.clip(action, low, high)

    def _warmup_action(self) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        if self.warmup_mode == "uniform" or self.warmup_action_raw is None:
            # A uniform warm-up still obeys the characterized safe physical
            # range.  This affects only warm-up; SAC's learned policy keeps
            # the full environment action space.
            action = np.random.uniform(
                self.warmup_uniform_low,
                np.asarray(self.env.action_space.high, dtype=np.float64),
            )
        else:
            action = self.warmup_action_raw.copy()
        if self.warmup_mode == "pink":
            noise = (
                self.warmup_noise.sample()
                if self.warmup_noise is not None
                else np.zeros(self.action_dim, dtype=np.float64)
            )
            # Keep initial replay inside the characterized warm-up envelope;
            # the learned policy still retains the full environment range.
            low = self.warmup_uniform_low
            high = np.asarray(self.env.action_space.high, dtype=np.float64)
            action = np.clip(action + self.warmup_noise_std * noise, low, high)
        else:
            # Preserve the historical behavior for configs that explicitly
            # used external pink noise during warm-up.
            noise = self._sample_exploration_noise()
            action = self._apply_exploration_noise(action, noise)
        return action, dict(
            log_prob=float("nan"),
            log_std=np.full(self.action_dim, np.nan, dtype=np.float64),
            action=np.asarray(action, dtype=np.float64).copy(),
            policy_action=np.asarray(action, dtype=np.float64).copy(),
            pink_noise=noise.copy(),
        )

    def evaluate_action(
        self,
        state: np.ndarray,
    ):
        return self.predict(state, evaluate=True)

    def estimate_value(
        self,
        state: np.ndarray,
        action: np.ndarray | None = None,
    ) -> float:
        """Return a deterministic critic value for lambda-discrepancy traces.

        The diagnostic needs one value estimate ``V(z_t)`` per transition.  We
        use the minimum twin-Q estimate at the policy mean action; this keeps
        the trace deterministic and avoids adding sampling noise to the
        representation comparison.  The method intentionally does not alter
        replay, optimizers, or recurrent state (lib4's current SAC is
        feed-forward over an explicitly constructed history vector).
        """
        state_tensor = self._state_tensor(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            if action is None:
                action_tensor = self._policy_mean_action_tensor(state_tensor)
            else:
                action_tensor = torch.as_tensor(
                    np.asarray(action, dtype=np.float32),
                    dtype=torch.float32,
                ).reshape(1, self.action_dim)
            q1, q2 = self.critic(state_tensor, action_tensor)
            value = torch.minimum(q1, q2)
        return float(value.detach().cpu().numpy().reshape(-1)[0])
    
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

        state_batch = self._normalize_state_tensor(state_batch)
        next_state_batch = self._normalize_state_tensor(next_state_batch)
        noise_state_batch = self._normalize_state_tensor(noise_state_batch)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha*next_log_prob
            # min_qf_target = torch.min(qf1_next_target, qf2_next_target)
            effective_gamma = self.gamma ** self.action_chunk_steps
            next_q_value = reward_batch + mask_batch * effective_gamma * min_qf_target
        
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

        pi, log_pi, pi_mean = self.policy.sample(state_batch)

        if self.caps_mode == "mean_action":
            # Regularizing independent SAC samples also penalizes the policy's
            # intended entropy.  CAPS instead acts on deterministic policy
            # means and averages one action-vector norm per replay sample.
            spatial_state = state_batch.clone()
            non_goal_dim = state_batch.shape[1] - int(self.env.dim_ref_traj)
            if self.caps_spatial_noise_normalized > 0.0 and non_goal_dim > 0:
                spatial_state[:, :non_goal_dim] += (
                    self.caps_spatial_noise_normalized
                    * torch.randn_like(spatial_state[:, :non_goal_dim])
                )
            next_mean = self._policy_mean_action_tensor(next_state_batch)
            spatial_mean = self._policy_mean_action_tensor(spatial_state)
            temporal_base = torch.linalg.vector_norm(
                pi_mean - next_mean,
                ord=2,
                dim=1,
            ).mean()
            spatial_base = torch.linalg.vector_norm(
                pi_mean - spatial_mean,
                ord=2,
                dim=1,
            ).mean()
        else:
            next_pi, _, _ = self.policy.sample(next_state_batch)
            noise_pi, _, _ = self.policy.sample(noise_state_batch)
            # Exact legacy reduction retained for old saved configurations.
            temporal_base = torch.norm(pi - next_pi, p=2)
            spatial_base = torch.norm(pi - noise_pi, p=2)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        pi_loss = ((self.alpha*log_pi) - min_qf_pi).mean()
        
        self.current_caps_factor = self._caps_factor()
        L_temporal = (
            self.current_caps_factor * self.temporal_weight * temporal_base
        )
        L_spatial = self.current_caps_factor * self.spatial_weight * spatial_base
        chunk_consistency_base = torch.zeros((), dtype=pi_mean.dtype, device=pi_mean.device)
        if (
            self.chunk_consistency_weight > 0.0
            and self.action_chunk_steps > 1
            and self.action_chunk_mode == "free"
        ):
            policy_sequence = pi_mean.reshape(
                -1,
                self.action_chunk_steps,
                int(self.env.dim_policy_act),
            )
            chunk_consistency_base = torch.mean(
                torch.square(policy_sequence[:, 1:] - policy_sequence[:, :-1])
            )
        L_chunk_consistency = self.chunk_consistency_weight * chunk_consistency_base
        policy_loss = pi_loss + L_temporal + L_spatial + L_chunk_consistency
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

            episode_index = self.last_epi + epi
            if (
                self.reset_after_warmup
                and episode_index == self.start_epi
                and hasattr(self.env, "request_hard_reset")
            ):
                self.env.request_hard_reset()
            state = self.env.reset()[0]

            epi_steps = 0
            total_critic_loss = 0
            total_policy_loss = 0
            update_count = 0
            episode_infos = []
            policy_diagnostics = []

            while epi_steps < self.horizon:
                in_warmup = self.start_epi > self.last_epi + epi
                if in_warmup:
                    action, warmup_diag = self._warmup_action()
                    policy_diagnostics.append(warmup_diag)
                else:
                    action, policy_diag = self.select_action_with_diagnostics(state)
                    policy_diagnostics.append(policy_diag)
                
                next_state, reward, done, _, info = self.env.step(action)
                episode_infos.append(info)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                can_update = self.update_during_warmup or not in_warmup
                if (
                    can_update
                    and len(self.buffer) > self.batch_size
                    and self.total_steps % self.update_every == 0
                ):
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
            global_epi = self.last_epi + epi

            clear_verbose = getattr(self.env, "clear_verbose", None)
            if callable(clear_verbose):
                clear_verbose()

            print(
                f'[ INFO] {self.model_name} ==> |epi|\t{self.last_epi + epi}\t'
                f'|step|\t{self.total_steps}\t'
                f'|reward|\t{epi_reward:.4f}'
            )
            
            if self.log:
                self.save_model(self.logger.model_path)
                if (
                    self.checkpoint_interval > 0
                    and global_epi % self.checkpoint_interval == 0
                ):
                    checkpoint_path = os.path.join(
                        self.logger.folder_path,
                        f"model_epi_{global_epi:04d}.pth",
                    )
                    self.save_model(checkpoint_path)
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
                diag = None
                if self.train_diag_interval > 0 and epi % self.train_diag_interval == 0:
                    diag = self._episode_diagnostics(
                        epi=global_epi,
                        reward=epi_reward,
                        infos=episode_infos,
                    )
                    self._append_train_diagnostics(diag)
                if policy_diagnostics:
                    self._append_exploration_diagnostics(
                        epi=global_epi,
                        diagnostics=policy_diagnostics,
                    )

                if (
                    self.train_verbose_interval > 0
                    and global_epi % self.train_verbose_interval == 0
                    and diag is not None
                ):
                    exploration = self._summarize_exploration(policy_diagnostics)
                    print(
                        f'[ DIAG] epi={global_epi} '
                        f'reward={diag.get("reward", epi_reward):.3f} '
                        f'pos_rmse={diag.get("pos_rmse", float("nan")):.3f} '
                        f'neg_rmse={diag.get("neg_rmse", float("nan")):.3f} '
                        f'disp_rmse={diag.get("disp_rmse", float("nan")):.3f} '
                        f'ch_pos_rate={diag.get("ch_pos_sign_change_per_sec", float("nan")):.2f} '
                        f'ch_neg_rate={diag.get("ch_neg_sign_change_per_sec", float("nan")):.2f} '
                        f'u_delta={self._format_action_delta(diag)} '
                        f'ctrl=({diag.get("u_pos_mean", float("nan")):.3f},'
                        f'{diag.get("u_neg_mean", float("nan")):.3f}) '
                        f'ctrl_rng=('
                        f'{diag.get("u_pos_min", float("nan")):.3f}-'
                        f'{diag.get("u_pos_max", float("nan")):.3f},'
                        f'{diag.get("u_neg_min", float("nan")):.3f}-'
                        f'{diag.get("u_neg_max", float("nan")):.3f}) '
                        f'raw=({diag.get("raw_u_pos_mean", float("nan")):.3f},'
                        f'{diag.get("raw_u_neg_mean", float("nan")):.3f}) '
                        f'alpha={exploration.get("alpha", float("nan")):.4f} '
                        f'log_std={exploration.get("log_std_mean", float("nan")):.3f} '
                        f'act_std={exploration.get("action_std_mean", float("nan")):.3f} '
                        f'sat={exploration.get("action_saturation_ratio", float("nan")):.3f}'
                    )

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

        # A K-step decision must be diagnosed at the physical-step rate.
        # Older code retained only the last substep and could therefore hide
        # within-chunk action/pressure oscillation.
        physical_infos: list[dict] = []
        for info in infos:
            substeps = info.get("chunk_substep_infos")
            if substeps:
                physical_infos.extend(substeps)
            else:
                physical_infos.append(info)
        infos = physical_infos

        # lib2 exposes chamber pressure directly (sen_pos/sen_neg), while
        # lib3 exposes actuator pressure as well.  Keep one diagnostics file
        # format per run instead of failing when the lib3-only fields are not
        # present.
        first_obs = infos[0].get("obs", {})
        if "sen_act_pos" not in first_obs:
            def values(key: str) -> np.ndarray:
                return np.asarray([info["obs"][key] for info in infos], dtype=np.float64)

            time_arr = values("curr_time")
            pos = values("sen_pos")
            neg = values("sen_neg")
            pos_ref = values("ref_pos")
            neg_ref = values("ref_neg")
            u_pos = values("ctrl_pos")
            u_neg = values("ctrl_neg")
            raw_action = np.asarray(
                [info.get("ctrl_input", [np.nan, np.nan]) for info in infos],
                dtype=np.float64,
            )

            def rmse(x: np.ndarray) -> float:
                return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0

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

            return dict(
                epi=int(epi),
                reward=float(reward),
                n=int(len(infos)),
                duration_sec=float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0,
                pos_rmse=rmse(pos_ref - pos),
                neg_rmse=rmse(neg_ref - neg),
                pos_bias=float(np.mean(pos_ref - pos)),
                neg_bias=float(np.mean(neg_ref - neg)),
                pos_sign_change_per_sec=sign_change_per_sec(pos),
                neg_sign_change_per_sec=sign_change_per_sec(neg),
                u_pos_mean=float(np.mean(u_pos)),
                u_neg_mean=float(np.mean(u_neg)),
                u_pos_min=float(np.min(u_pos)),
                u_pos_max=float(np.max(u_pos)),
                u_neg_min=float(np.min(u_neg)),
                u_neg_max=float(np.max(u_neg)),
                raw_u_pos_mean=float(np.nanmean(raw_action[:, 0])),
                raw_u_neg_mean=float(np.nanmean(raw_action[:, 1])),
                raw_u_pos_min=float(np.nanmin(raw_action[:, 0])),
                raw_u_pos_max=float(np.nanmax(raw_action[:, 0])),
                raw_u_neg_min=float(np.nanmin(raw_action[:, 1])),
                raw_u_neg_max=float(np.nanmax(raw_action[:, 1])),
                u_pos_delta=float(np.mean(np.abs(np.diff(u_pos)))) if len(u_pos) > 1 else 0.0,
                u_neg_delta=float(np.mean(np.abs(np.diff(u_neg)))) if len(u_neg) > 1 else 0.0,
            )

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

        diagnostics = dict(
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
        if "sen_env_displacement" in first_obs and "ref_displacement" in first_obs:
            displacement = arr(("obs", "sen_env_displacement"))
            displacement_ref = arr(("obs", "ref_displacement"))
            displacement_err = displacement_ref - displacement
            ref_delta = np.diff(displacement_ref)
            displacement_delta = np.diff(displacement)
            moving = np.abs(ref_delta) > 1e-9
            direction_agreement = (
                float(np.mean(np.sign(ref_delta[moving]) == np.sign(displacement_delta[moving])))
                if bool(np.any(moving))
                else 0.0
            )

            def reward_sum(key: str) -> float:
                return float(
                    np.sum(
                        [info.get("reward", {}).get(key, 0.0) for info in infos],
                        dtype=np.float64,
                    )
                )

            diagnostics.update(
                disp_rmse=rmse(displacement_err),
                disp_bias=float(np.mean(displacement_err)),
                disp_min=float(np.min(displacement)),
                disp_max=float(np.max(displacement)),
                disp_span=float(np.ptp(displacement)),
                disp_ref_min=float(np.min(displacement_ref)),
                disp_ref_max=float(np.max(displacement_ref)),
                disp_ref_span=float(np.ptp(displacement_ref)),
                disp_direction_agreement=direction_agreement,
                pos_tracking_reward_sum=reward_sum("pos_curr_reward"),
                neg_tracking_reward_sum=reward_sum("neg_curr_reward"),
                disp_tracking_reward_sum=reward_sum("disp_curr_reward"),
            )
        return diagnostics

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

    @staticmethod
    def _format_action_delta(diag: dict[str, float]) -> str:
        """Return a compact action-change summary for the console."""
        keys = [key for key in diag if key.endswith("_delta")]
        if not keys:
            return "nan"
        return f'{float(np.mean([diag[key] for key in keys])):.4f}'

    def _summarize_exploration(
        self,
        diagnostics: list[dict[str, np.ndarray | float]],
    ) -> dict[str, float]:
        """Summarize policy sampling without adding per-step console output."""
        if not diagnostics:
            return {}
        log_stds = np.stack([d["log_std"] for d in diagnostics], axis=0)
        actions = np.stack([d["action"] for d in diagnostics], axis=0)
        pink = np.stack([d["pink_noise"] for d in diagnostics], axis=0)
        alpha = self.alpha.item() if torch.is_tensor(self.alpha) else float(self.alpha)
        return dict(
            alpha=float(alpha),
            log_std_mean=float(np.mean(log_stds)),
            action_std_mean=float(np.mean(np.std(actions, axis=0))),
            action_saturation_ratio=float(
                np.mean((actions <= 0.01) | (actions >= 0.99))
            ),
            pink_noise_abs_mean=float(np.mean(np.abs(pink))),
            pink_noise_std=float(np.std(pink)),
        )

    def _append_exploration_diagnostics(
        self,
        epi: int,
        diagnostics: list[dict[str, np.ndarray | float]],
    ) -> None:
        """Persist policy-side exploration statistics per episode."""
        if not diagnostics:
            return
        log_probs = np.asarray([d["log_prob"] for d in diagnostics], dtype=np.float64)
        log_stds = np.stack([d["log_std"] for d in diagnostics], axis=0)
        actions = np.stack([d["action"] for d in diagnostics], axis=0)
        pink = np.stack([d["pink_noise"] for d in diagnostics], axis=0)
        alpha = self.alpha.item() if torch.is_tensor(self.alpha) else float(self.alpha)
        row = dict(
            epi=int(epi),
            alpha=float(alpha),
            neg_log_prob_mean=float(np.mean(-log_probs)),
            log_std_mean=float(np.mean(log_stds)),
            log_std_min=float(np.min(log_stds)),
            log_std_max=float(np.max(log_stds)),
            action_std_mean=float(np.mean(np.std(actions, axis=0))),
            action_saturation_ratio=float(
                np.mean((actions <= 0.01) | (actions >= 0.99))
            ),
            pink_noise_abs_mean=float(np.mean(np.abs(pink))),
            pink_noise_std=float(np.std(pink)),
        )
        path = f"{self.logger.folder_path}/exploration_diagnostics.csv"
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
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
