import torch
from torch import nn
from torchlake.common.models import (
    DepthwiseSeparableConv2d,
    ResBlock,
    SqueezeExcitation2d,
)
from torchvision.ops import Conv2dNormActivation


class DepthwiseSeparableConv2dV3(DepthwiseSeparableConv2d):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        enable_bn: tuple[bool, bool] = (True, True),
        enable_relu: bool = False,
        enable_se: bool = True,
        reduction_ratio: int = 4,
    ):
        """Squeeze and excitation depthwise separable convolution [1905.02244]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            padding (tuple[int], optional): padding of both layers. Defaults to (0, 0).
            dilation (tuple[int], optional): dilation of both layers. Defaults to (1, 1).
            enable_bn (tuple[bool, bool], optional): enable_bn of both layers. Defaults to (True, True).
            enable_relu (bool, optional): enable relu, otherwise hard-swish. Defaults to False.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        activations = (
            (nn.ReLU(True), nn.ReLU(True))
            if enable_relu
            else (nn.Hardswish(), nn.Identity())
        )

        super().__init__(
            input_channel,
            output_channel,
            kernel,
            stride,
            padding,
            dilation,
            enable_bn,
            activations,
        )

        self.se = (
            SqueezeExcitation2d(
                input_channel,
                reduction_ratio=reduction_ratio,
                activations=(nn.ReLU(True), nn.Hardsigmoid()),
            )
            if enable_se
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise_layer(x)
        y = self.se(y)
        y = self.pointwise_layer(y)

        return y


class LinearBottleneckV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        expansion_size: int = 1,
        enable_relu: bool = False,
        enable_se: bool = True,
        reduction_ratio: int = 4,
    ):
        """Linear bottleneck for mobilenet v3 [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            enable_relu (bool, optional): enable relu, otherwise hard-swish. Defaults to False.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        super().__init__()
        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                expansion_size,
                1,
                activation_layer=nn.ReLU if enable_relu else nn.Hardswish,
                inplace=enable_relu,
            ),
            DepthwiseSeparableConv2dV3(
                expansion_size,
                output_channel,
                kernel,
                stride=stride,
                padding=kernel // 2,
                enable_se=enable_se,
                reduction_ratio=reduction_ratio,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvertedResidualBlockV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        expansion_size: int = 1,
        enable_relu: bool = False,
        enable_se: bool = True,
        reduction_ratio: int = 4,
    ):
        """Inverted residual block for mobilenet v3 [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            activation (tuple[nn.Module  |  None], optional): activation of both layers. Defaults to nn.Hardswish().
            enable_relu (bool, optional): enable relu, otherwise hard-swish. Defaults to False.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        super().__init__()
        layer = LinearBottleneckV3(
            input_channel,
            output_channel,
            kernel,
            stride,
            expansion_size,
            enable_relu=enable_relu,
            enable_se=enable_se,
            reduction_ratio=reduction_ratio,
        )
        self.layer = (
            ResBlock(
                input_channel,
                output_channel,
                layer,
                activation=None,
            )
            if input_channel == output_channel and stride == 1
            else layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
