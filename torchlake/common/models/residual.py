from typing import Literal

import torch
from torch import nn

from torchlake.common.models.conv import ConvBNReLU


class ResBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        block: nn.Module,
        stride: int = 1,
        activation: nn.Module | None = nn.ReLU(True),
        shortcut: nn.Module | None = None,
        dimension: Literal["1d", "2d", "3d"] = "2d",
    ):
        """residual block
        skip connection is 1x1 conv shortcut if input_channel != output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            block (nn.Module): block class
            stride (int, optional): stride of identity mapping. Defaults to 1.
            activation (tuple[nn.Module  |  None], optional): activation of residual output. Defaults to nn.ReLU(True).
            shortcut (nn.Module, optional): shortcut class. Defaults to None.
            dimension (Literal["1d"] | Literal["2d"] | Literal["3d"], optional): 1d, 2d or 3d. Defaults to "2d".
        """
        super().__init__()
        self.activation = activation

        self.block = block

        self.downsample = shortcut or self.build_shortcut(
            input_channel,
            output_channel,
            stride,
            dimension,
        )

    def build_shortcut(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        dimension: Literal["1d", "2d", "3d"] = "2d",
    ) -> nn.Module:
        """build shortcut

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            dimension (Literal["1d"] | Literal["2d"] | Literal["3d"], optional): 1d, 2d or 3d. Defaults to "2d".
        """

        if input_channel == output_channel and stride == 1:
            return nn.Identity()

        layer = nn.Sequential(
            ConvBNReLU(
                input_channel,
                output_channel,
                1,
                stride=stride,
                enable_bn=None,
                activation=None,
                dimension=dimension,
            )
        )

        return layer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self.block(x, *args, **kwargs) + self.downsample(x)

        if self.activation is None:
            return y
        else:
            return self.activation(y)
