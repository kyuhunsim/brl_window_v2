from typing import Tuple, Optional
from collections import deque
import random

import numpy as np
import torch

import pickle
import os
import shutil

class ReplayBuffer():
    def __init__(
        self,
        capacity: int = 1e6,
        noise_std: float = 0.1,
        dim_goal: Optional[int] = None
    ):
        self.buffer = deque(maxlen=int(capacity))
        self.noise_std = noise_std
        self.dim_goal = dim_goal
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        mask = 0 if done else 1
        goal = state[-self.dim_goal:]
        obs = state[:-self.dim_goal]
        self.buffer.append((obs, goal, action, reward, next_state, mask))
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        obs, goal, action, reward, next_state, done = map(np.stack, zip(*batch))

        # State ==> obs + Goal
        state = np.c_[obs, goal]

        noise = np.random.normal(loc=0, scale=self.noise_std, size=obs.shape)
        noise_obs = obs + noise
        noise_state = np.c_[noise_obs, goal]

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
            torch.tensor(noise_state, dtype=torch.float32)
        )
        
    def __len__(self):
        return len(self.buffer)
    
    def save_buffer(
        self,
        path: str = 'buffer_sac.pkl'
    ) -> None:
        path = os.fspath(path)
        with open(f'{path}.tmp', 'wb') as f:
            pickle.dump(self.buffer, f)
        os.replace(f'{path}.tmp', path)
        backup = f'{path[0:-4]}_backup.pkl'
        shutil.copyfile(path, f'{backup}.tmp')
        os.replace(f'{backup}.tmp', backup)

    def load_buffer(
        self,
        path: str
    ) -> None:
        try:
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
        except:
            with open(f'{path[0:-4]}_backup.pkl', 'rb') as f:
                self.buffer = pickle.load(f)
    
    def clear_buffer(self):
        self.buffer.clear()
