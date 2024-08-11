import torch
from torch import nn
from torchlake.common.models import (
    ResBlock,
    SqueezeExcitation2d,
)
from torchlake.common.mixins.network import SeMixin
from torchvision.ops import Conv2dNormActivation


class GhostModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        s: int = 2,
        d: int = 1,
    ):
        """GhostNet ghost module [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            s (int): num of ghost features, split ratio of output features for depthwise convolution. Defaults to 2.
            d (int): kernel size of depthwise convolution layer. Defaults to 1.
        """
        super(GhostModule, self).__init__()
        self.identity_channel = int(output_channel / s)
        transformed_channel = output_channel - self.identity_channel

        self.pointwise_conv = Conv2dNormActivation(
            input_channel,
            output_channel,
            1,
            norm_layer=None,
            activation_layer=None,
        )

        self.ghost = Conv2dNormActivation(
            transformed_channel,
            transformed_channel,
            d,
            groups=transformed_channel,
            norm_layer=None,
            activation_layer=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c // s, h, w
        y = self.pointwise_conv(x)

        # b, c / s + c * (s-1) / s, h, w
        return torch.cat(
            [
                y[:, : self.identity_channel, :, :],
                self.ghost(y[:, self.identity_channel :, :, :]),
            ],
            1,
        )


class GhostBottleNeck(SeMixin, nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        s: int = 2,
        d: int = 1,
        expansion_size: int = 1,
        enable_se: bool = True,
        reduction_ratio: int = 4,
    ):
        """GhostNet bottleneck [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            s (int): num of ghost features, group parameter. Defaults to 2.
            d (int): kernel size of depthwise convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        super(GhostBottleNeck, self).__init__()

        # basic block
        self.block = self.build_block(
            input_channel,
            output_channel,
            kernel,
            stride,
            s,
            d,
            expansion_size,
        )

        self.se = (
            SqueezeExcitation2d(
                output_channel,
                reduction_ratio,
            )
            if enable_se
            else nn.Identity()
        )

    def build_block(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        s: int = 2,
        d: int = 1,
        expansion_size: int = 1,
    ) -> nn.Module:
        """Block in GhostNet bottleneck [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            s (int): num of ghost features, group parameter. Defaults to 2.
            d (int): kernel size of depthwise convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
        """
        block = [
            GhostModule(
                input_channel,
                expansion_size,
                s=s,
                d=d,
            ),
            nn.BatchNorm2d(expansion_size),
            nn.ReLU(inplace=True),
            GhostModule(
                expansion_size,
                output_channel,
                s=s,
                d=d,
            ),
            nn.BatchNorm2d(output_channel),
        ]

        if stride > 1:
            block.insert(
                -3,
                Conv2dNormActivation(
                    expansion_size,
                    expansion_size,
                    kernel,
                    stride,
                    groups=expansion_size,
                    activation_layer=None,
                ),
            )

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return self.se(y)


class GhostLayer(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        s: int = 2,
        d: int = 1,
        expansion_size: int = 1,
        enable_se: bool = True,
        reduction_ratio: int = 4,
    ):
        """GhostNet layer [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            s (int): num of ghost features, group parameter. Defaults to 2.
            d (int): kernel size of depthwise convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
        """
        super(GhostLayer, self).__init__()

        block = GhostBottleNeck(
            input_channel,
            output_channel,
            kernel,
            stride,
            s,
            d,
            expansion_size,
            enable_se,
            reduction_ratio,
        )

        # packed into resblock
        self.block = ResBlock(
            input_channel,
            output_channel,
            block,
            stride,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
