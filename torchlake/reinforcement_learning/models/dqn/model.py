import random

import torch
from torch import nn

from .network import QNetwork


class Dqn(nn.Module):
    def __init__(self, n_actions: int, device: torch.device):
        super(Dqn, self).__init__()
        self.n_actions = n_actions
        self.device = device

        self.target_net = QNetwork(n_actions)

    def update_target_net(self, state_dict: dict):
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_net(x)

    def get_action(self, state: torch.Tensor, epsilon: float = 1e-2) -> torch.Tensor:
        if random.random() > epsilon:
            with torch.no_grad():
                # pick action with the largest expected reward.
                return self.target_net(state).argmax(-1, keepdim=True)
        else:
            return torch.LongTensor(
                [[random.randrange(self.n_actions)] for _ in range(state.size(0))]
            ).to(self.device)
