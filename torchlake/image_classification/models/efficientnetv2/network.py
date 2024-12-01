import torch
from torch import nn
from torchlake.common.models import ResBlock, SqueezeExcitation2d
from torchvision.ops import Conv2dNormActivation


class FusedMBConv(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        expansion_size: int = 1,
        enable_se: bool = True,
        reduction_ratio: int = 4,
    ):
        """Fused Mobile Inverted Residual Bottleneck [2104.00298v3]

        Args:
            input_channel (int): input channel size.
            output_channel (int): output channel size.
            kernel (int, optional): kernel size. Defaults to 3.
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        super().__init__()
        self.expansion_layer = Conv2dNormActivation(
            input_channel,
            expansion_size,
            kernel,
            stride=stride,
            activation_layer=nn.SiLU,
            inplace=False,
        )

        self.se = (
            SqueezeExcitation2d(
                expansion_size,
                reduction_ratio,
                activations=(nn.SiLU(), nn.Sigmoid()),
            )
            if enable_se
            else nn.Identity()
        )

        self.contraction_layer = Conv2dNormActivation(
            expansion_size,
            output_channel,
            kernel,
            stride=stride,
            activation_layer=nn.SiLU,
            inplace=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expansion_layer(x)
        y = self.se(y)
        return self.contraction_layer(y)


class InvertedResidualBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        expansion_size: int = 1,
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
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        super().__init__()
        layer = FusedMBConv(
            input_channel,
            output_channel,
            kernel=kernel,
            stride=stride,
            expansion_size=expansion_size,
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
