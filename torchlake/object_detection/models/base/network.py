import torch
from torch import nn
import torch.nn.functional as F

from torchlake.common.network import ConvBnRelu


class ConvBlock(nn.Module):
    """ConvBNReLU block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple[int],
        stride: int | tuple[int] = 1,
        enable_bn: bool = True,
        activation: nn.Module | None = nn.LeakyReLU(0.1),
    ):
        super(ConvBlock, self).__init__()
        self.conv = ConvBnRelu(
            in_channels,
            out_channels,
            kernel,
            padding=(
                (k // 2 for k in kernel) if hasattr(kernel, "__iter__") else kernel // 2
            ),
            stride=stride,
            enable_bn=enable_bn,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RegHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        num_priors: int,
        coord_dims: int = 4,
    ):
        super(RegHead, self).__init__()
        self.loc = nn.Conv2d(
            input_channel,
            num_priors * coord_dims,
            kernel_size=3,
            padding=1,
        )
        self.conf = nn.Conv2d(
            input_channel,
            num_priors * num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loc(x), self.conf(x)
