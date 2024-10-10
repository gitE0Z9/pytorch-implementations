import torch
from torch import nn

from torchlake.common.models import L2Norm


class GlobalContextModule(nn.Module):

    def __init__(self, input_channel: int):
        """Global context module in ParseNet [1506.04579v2]

        Args:
            input_channel (int): _description_
        """
        super().__init__()
        self.norm = L2Norm(input_channel, scale=1.0)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            L2Norm(input_channel, scale=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.norm(x), self.pool(x).repeat(1, 1, *x.shape[2:])], 1)
