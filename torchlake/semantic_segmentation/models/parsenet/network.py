import torch
from torch import nn

from torchlake.common.models import L2Norm


class GlobalContextModule(nn.Module):

    def __init__(self, input_channel: int, output_channel: int):
        """Global context module in ParseNet [1506.04579v2]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
        """
        super().__init__()
        self.norm = nn.Sequential(
            L2Norm(input_channel, scale=1.0),
            nn.Conv2d(input_channel, output_channel, 1),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            L2Norm(input_channel, scale=1.0),
            nn.Conv2d(input_channel, output_channel, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.pool(x).repeat(1, 1, *x.shape[2:])
