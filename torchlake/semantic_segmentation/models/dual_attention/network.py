"""Reference from https://github.com/junfu1115/DANet/blob/master/encoding/models/sseg/danet.py"""

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class SpatialAttention2d(nn.Module):
    """Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim: int):
        super(SpatialAttention2d, self).__init__()

        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1,
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1,
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1,
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H*W
        q = self.query_conv(x).flatten(-2)
        # B, C, H*W
        k = self.key_conv(x).flatten(-2)
        # B, C, H*W
        v = self.value_conv(x).flatten(-2)

        # q.T @ k
        att = torch.einsum("bck, bcs -> bks", q, k)
        att = att.softmax(-1)

        # v @ a.T
        out = torch.einsum("bcs, bks -> bck", v, att)
        out = out.view(x.size())

        return self.alpha * out + x


class ChannelAttention2d(nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(ChannelAttention2d, self).__init__()

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        # B, C, H*W
        q = k = v = x.flatten(-2)

        # q @ k.T
        att = torch.einsum("bcs, bks -> bck", q, k)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        att = att.softmax(-1)

        # a @ v
        out = torch.einsum("bkc, bcs -> bks", att, v)
        out = out.view(x.size())

        return self.beta * out + x


class DualAttention2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        reduction_ratio: float = 1,
    ):
        super(DualAttention2d, self).__init__()
        inter_channel = input_channel // reduction_ratio

        self.sa_conv = nn.Sequential(
            Conv2dNormActivation(input_channel, inter_channel),
            SpatialAttention2d(inter_channel),
            Conv2dNormActivation(inter_channel, inter_channel),
        )
        self.ca_conv = nn.Sequential(
            Conv2dNormActivation(input_channel, inter_channel),
            ChannelAttention2d(),
            Conv2dNormActivation(inter_channel, inter_channel),
        )

    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca_conv(x)

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa_conv(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.sa_conv(x) + self.ca_conv(x)
