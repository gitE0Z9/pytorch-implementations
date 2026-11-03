import torch
from torch import nn
from torchlake.common.models import ConvBNReLU


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        group: int = 32,
        expansion_factor: int = 2,
        pre_activation: bool = False,
    ):
        """convolution block in resnext
        kernel: 3x3 -> 3x3
        channel: input_channel -> output_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            group (int, optional): group of middle layer. Defaults to 32.
            expansion_factor (int, optional): hidden dimension = expansion factor x output channel size. Defaults to 2.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(
                input_channel,
                output_channel * expansion_factor,
                3,
                padding=1,
                stride=stride,
                group=group,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                output_channel * expansion_factor,
                output_channel,
                3,
                padding=1,
                group=group,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        group: int = 32,
        expansion_factor: int = 2,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnext
        kernel: 1x1 -> 3x3 -> 1x1
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            group (int, optional): group of middle layer. Defaults to 32.
            expansion_factor (int, optional): expansion factor of base channel. Defaults to 2.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(
                input_channel,
                hidden_dim,
                1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                group=group,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                hidden_dim,
                hidden_dim * expansion_factor,
                1,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
