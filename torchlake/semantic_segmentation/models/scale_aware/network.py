from typing import Sequence
import torch
from torch import nn


class ScaleAwareAttention(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int = 512,
        num_scales: int = 3,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_scales, 1),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.layers(torch.cat(features, 1)).softmax(1)
