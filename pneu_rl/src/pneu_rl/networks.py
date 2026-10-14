from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from gymnasium import spaces

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_value = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
    
    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.linear_value(x)

        return value

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_space: Optional[spaces.Box] = None,
        log_std_min: float = -1.6,
        log_std_max: float = 0,
    ):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, action_dim)
        self.linear_log_std = nn.Linear(hidden_dim, action_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1., dtype=torch.float64).detach()
            self.action_bias = torch.tensor(0., dtype=torch.float64).detach()
        else:
            low, high = action_space.low, action_space.high
            self.action_scale = torch.tensor((high - low)/2., dtype=torch.float64).detach()
            self.action_bias = torch.tensor((high + low)/2., dtype=torch.float64).detach()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        low, high = log_std_min, log_std_max
        self.log_std_scale = torch.tensor((high - low)/2., dtype=torch.float64).detach()
        self.log_std_bias = torch.tensor((high + low)/2., dtype=torch.float64).detach()

        self.apply(weights_init_)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.action_scale*F.tanh(self.linear_mean(x)) + self.action_bias
        log_std = self.log_std_scale*F.tanh(self.linear_log_std(x)) + self.log_std_bias

        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor
    ):
        mean, log_std = self.forward(state)

        dist = Normal(mean, log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return mean, action, log_prob
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ):
        mean, log_std = self.forward(states)

        dist = Normal(mean, log_std.exp())
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return log_probs, entropies


class SACPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_space: Optional[spaces.Box] = None,
        log_std_min: float = -1.6,
        log_std_max: float = -1.6,
        initial_action: Optional[np.ndarray] = None,
    ):
        super(SACPolicy, self).__init__()
        
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, action_dim)
        self.linear_log_std = nn.Linear(hidden_dim, action_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1., dtype=torch.float32).detach()
            self.action_bias = torch.tensor(0., dtype=torch.float32).detach()
        else:
            self.action_scale = torch.tensor(
                (action_space.high - action_space.low)/2.,
                dtype = torch.float32
            ).detach()
            self.action_bias = torch.tensor(
                (action_space.high + action_space.low)/2.,
                dtype = torch.float32
            ).detach()
        
        log_std_scale = (log_std_max - log_std_min)/2.
        log_std_bias = (log_std_max + log_std_min)/2.
        self.log_std_scale = log_std_scale*torch.ones(action_dim)
        self.log_std_bias = log_std_bias*torch.ones(action_dim)

        if initial_action is not None:
            initial = np.asarray(initial_action, dtype=np.float32).reshape(-1)
            if initial.size == 1:
                initial = np.full(action_dim, float(initial[0]), dtype=np.float32)
            if initial.size != action_dim:
                raise ValueError(
                    "initial_action must be scalar or match action dimension: "
                    f"got {initial.size} for {action_dim}"
                )
            low = np.asarray(action_space.low if action_space is not None else np.zeros(action_dim), dtype=np.float32)
            high = np.asarray(action_space.high if action_space is not None else np.ones(action_dim), dtype=np.float32)
            initial = np.clip(initial, low, high)
            normalized = 2.0 * (initial - low) / np.maximum(high - low, 1e-6) - 1.0
            normalized = np.clip(normalized, -0.999, 0.999)
            bias = np.arctanh(normalized)
            with torch.no_grad():
                # Start from the requested operating point for every state;
                # otherwise random hidden-to-mean weights can overwhelm the
                # bias before the first policy update.
                self.linear_mean.weight.zero_()
                self.linear_mean.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear_1(state))
        x = F.relu(self.linear_2(x))
        mean = self.linear_mean(x)
        log_std = self.linear_log_std(x)
        log_std = self.log_std_scale*log_std.tanh() + self.log_std_bias

        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
        eps: float = 1e-6
    ):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        
        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)

        action = self.action_scale*a + self.action_bias
        mean = self.action_scale*torch.tanh(mean) + self.action_bias

        log_prob = dist.log_prob(u)
        log_prob -= torch.log(self.action_scale*(1 - a.pow(2)) + eps)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean
        
class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super(QNetwork, self).__init__()

        self.linear_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q1 = nn.Linear(hidden_dim, 1)

        self.linear_3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, action], 1)

        x1 = F.relu(self.linear_1(x))
        x1 = F.relu(self.linear_2(x1))
        q1 = self.linear_q1(x1)

        x2 = F.relu(self.linear_3(x))
        x2 = F.relu(self.linear_4(x2))
        q2 = self.linear_q2(x2)

        return q1, q2


    
