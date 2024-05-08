import torch
from torch import nn
from torchlake.common.models import DepthwiseSeparableConv2d, ResBlock
from torchlake.common.network import ConvBnRelu


class LinearBottleneck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        expansion_ratio: int = 1,
    ):
        """Linear bottleneck [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_ratio (int, optional): expansion ratio. Defaults to 1.
        """
        super(LinearBottleneck, self).__init__()
        self.layers = nn.Sequential(
            ConvBnRelu(
                input_channel,
                expansion_ratio * input_channel,
                1,
                activation=nn.ReLU6(),
            ),
            DepthwiseSeparableConv2d(
                expansion_ratio * input_channel,
                output_channel,
                stride=stride,
                activation=(nn.ReLU6(), nn.Identity()),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvertedResidualBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        expansion_ratio: int = 1,
    ):
        """Inverted residual block [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_ratio (int, optional): expansion ratio. Defaults to 1.
        """
        super(InvertedResidualBlock, self).__init__()
        layer = LinearBottleneck(
            input_channel,
            output_channel,
            stride,
            expansion_ratio,
        )
        self.layer = (
            ResBlock(
                input_channel,
                output_channel,
                layer,
                activation=None,
            )
            if stride == 1
            else layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SeLinearBottleneck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        expansion_size: int = 1,
        reduction_ratio=16,
    ):
        """Squeeze and excitation bottleneck [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
        """
        super(SeLinearBottleneck, self).__init__()
        self.layers = nn.Sequential(
            ConvBnRelu(
                input_channel,
                expansion_size,
                1,
                activation=nn.Hardswish(),
            ),
            DepthwiseSeparableConv2d(
                expansion_size,
                output_channel,
                kernel,
                stride=stride,
                expansion_size=expansion_size,
                activation=(nn.Hardswish(), nn.ReLU6()),
                enable_se=True,
                reduction_ratio=reduction_ratio,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SeInvertedResidualBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        expansion_size: int = 1,
        activation: nn.Module | None = nn.Hardswish(),
        enable_se: bool = True,
        reduction_ratio=16,
    ):
        """Squeeze and excitation inverted residual block [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            activation (tuple[nn.Module  |  None], optional): activation of both layers. Defaults to nn.Hardswish().
        """
        super(SeInvertedResidualBlock, self).__init__()
        layer = SeLinearBottleneck(
            input_channel,
            output_channel,
            kernel,
            stride,
            expansion_size,
            reduction_ratio,
        )
        self.layer = (
            ResBlock(
                input_channel,
                output_channel,
                layer,
                activation=None,
            )
            if stride == 1
            else layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
