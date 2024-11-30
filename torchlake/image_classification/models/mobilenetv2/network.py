import torch
from torch import nn
from torchlake.common.models import (
    DepthwiseSeparableConv2d,
    ResBlock,
)
from torchvision.ops import Conv2dNormActivation


class LinearBottleneck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        expansion_ratio: int = 1,
    ):
        """Linear bottleneck [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int, optional): kernel size of depthwise separable convolution layer. Defaults to 3.
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_ratio (int, optional): expansion ratio. Defaults to 1.
        """
        super().__init__()
        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                expansion_ratio * input_channel,
                1,
                activation_layer=nn.ReLU6,
                inplace=False,
            ),
            DepthwiseSeparableConv2d(
                expansion_ratio * input_channel,
                output_channel,
                kernel=kernel,
                stride=stride,
                padding=kernel // 2,
                activations=(nn.ReLU6(), nn.Identity()),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvertedResidualBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        expansion_ratio: int = 1,
    ):
        """Inverted residual block [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int, optional): kernel size of depthwise separable convolution layer. Defaults to 3.
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_ratio (int, optional): expansion ratio. Defaults to 1.
        """
        super().__init__()
        layer = LinearBottleneck(
            input_channel,
            output_channel,
            kernel=kernel,
            stride=stride,
            expansion_ratio=expansion_ratio,
        )
        self.layer = (
            ResBlock(
                input_channel,
                output_channel,
                layer,
                activation=None,
            )
            if input_channel == output_channel
            else layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
