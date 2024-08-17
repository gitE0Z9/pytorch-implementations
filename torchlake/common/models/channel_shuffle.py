import torch
from torch import nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int = 1):
        """Channel shuffle layer in paper [1707.01083v2]

        Args:
            groups (int, optional): number of groups of channels. Defaults to 1.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return (
            x.view(b, self.groups, c // self.groups, h, w)
            .transpose(1, 2)
            .reshape(b, c, h, w)
        )
