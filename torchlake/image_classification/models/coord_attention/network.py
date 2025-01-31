import torch
from torch import nn
from torchlake.common.mixins.network import SEMixin
from torchvision.ops import Conv2dNormActivation

from ..resnet.network import BottleNeck


class CoordinateAttention2d(nn.Module):

    def __init__(self, in_dim: int, reduction_ratio: float = 1):
        """Coordinate attention module

        Args:
            in_dim (int): input dimension
            reduction_ratio (float, optional): reduction ratio. Defaults to 1.
        """
        super().__init__()
        compressed_channel = in_dim // reduction_ratio
        self.conv = Conv2dNormActivation(in_dim, compressed_channel, 1)
        self.conv_x = nn.Conv2d(compressed_channel, in_dim, 1)
        self.conv_y = nn.Conv2d(compressed_channel, in_dim, 1)

    def get_x_poolings(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(3, keepdim=True)

    def get_y_poolings(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(2, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        # B, C, H, 1
        # B, C, 1, W
        x_pooled, y_pooled = self.get_x_poolings(x), self.get_y_poolings(x)
        # B, C, H+W, 1
        y = torch.cat([x_pooled, y_pooled.transpose(-2, -1)], 2)
        # B, C/r, H+W, 1
        y = self.conv(y)

        # B, C/r, H, 1
        # B, C/r, W, 1
        x_attention, y_attention = torch.split(y, [h, w], 2)

        # B, C, H, 1
        # B, C, 1, W
        x_attention, y_attention = (
            self.conv_x(x_attention).sigmoid(),
            self.conv_y(y_attention).transpose(-2, -1).sigmoid(),
        )

        return x * x_attention * y_attention


class BottleNeck(SEMixin, BottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """bottleneck block in coordinate attention resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before transformation [1603.05027v3]
        """
        super().__init__(
            input_channel,
            block_base_channel,
            stride,
            pre_activation,
        )
        self.se = CoordinateAttention2d(block_base_channel * 4, 32)
