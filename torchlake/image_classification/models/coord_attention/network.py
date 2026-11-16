import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.mixins.network import SEMixin

from ..resnet.network import BottleNeck as BottleNeckBase


class CoordinateAttention2d(nn.Module):

    def __init__(self, input_channel: int, reduction_ratio: float = 1):
        """Coordinate attention module

        Args:
            input_channel (int): input dimension
            reduction_ratio (float, optional): reduction ratio. Defaults to 1.
        """
        super().__init__()
        compressed_channel = input_channel // reduction_ratio
        self.conv = Conv2dNormActivation(input_channel, compressed_channel, 1)
        self.conv_x = nn.Conv2d(compressed_channel, input_channel, 1)
        self.conv_y = nn.Conv2d(compressed_channel, input_channel, 1)

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


class BottleNeck(SEMixin, BottleNeckBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        pre_activation: bool = False,
        reduction_ratio: int = 32,
    ):
        """bottleneck block in coordinate attention resnet
        kernel: 1x1 -> 3x3 -> 1x1 -> coordinate attention
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 4.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            reduction_ratio (int, optional): reduction ratio of coordinate attention. Defaults to 32.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        self.se = CoordinateAttention2d(hidden_dim * expansion_factor, reduction_ratio)
