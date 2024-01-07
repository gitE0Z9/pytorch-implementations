from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import ReplayMemory, Transition


class AtariDataset(Dataset):
    def __init__(
        self,
        id: str,
        capacity: int,
        batch_size: int,
        transform=None,
        render_mode: Literal["human", "rgb_array", "ram", "gray"] = "rgb_array",
    ):
        # id : https://gymnasium.farama.org/environments/atari/complete_list/
        self.batch_size = batch_size
        self.capacity = capacity
        self.replays = ReplayMemory(capacity)
        self.transform = transform
        self.env = gym.make(id, render_mode=render_mode)

    def __len__(self):
        return np.Inf

    @property
    def n_actions(self) -> int:
        return self.env.action_space.n

    def init(self) -> tuple[np.ndarray, dict[str, int]]:
        init_screen, info = self.env.reset()
        return init_screen, info

    def get_state(self) -> torch.Tensor | np.ndarray:
        screen = self.env.render()
        if self.transform:
            screen = self.transform(screen)

        return screen

    # def get_last_replays(self):
    #     self.replays.memory.

    def __getitem__(self, idx: int):
        if idx == 0:
            self.init()
        return self.get_state()

    def get_next_step(self, action: torch.Tensor):
        # observation, reward, terminated, truncated, info
        next_state, reward, done, _, _ = self.env.step(action.item())

        return next_state, reward, done

    def push_memory(self, transition: Transition):
        self.replays.push(transition)

    def close(self):
        self.env.reset()
        self.env.close()
