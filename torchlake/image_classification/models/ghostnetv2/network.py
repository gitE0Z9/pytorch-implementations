import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.models import ResBlock, SqueezeExcitation2d
from torchvision.ops import Conv2dNormActivation

from ..ghostnet.network import GhostModule


class DFCAttention(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        horizontal_kernel: int = 3,
        vertical_kernel: int = 3,
    ):
        """Decoupled fully connected (DFC) attention in paper [2211.12905v1]

        Args:
            input_channel (int):input channel size.
            output_channel (int, optional): output channel size. Defaults to 1.
            horizontal_kernel (int, optional): horizontal attention kernel size. Defaults to 3.
            vertical_kernel (int, optional): vertical attention kernel size. Defaults to 3.
        """
        super().__init__()
        self.blocks = nn.Sequential(
            nn.AvgPool2d(2, 2),
            Conv2dNormActivation(
                input_channel,
                output_channel,
                1,
                activation_layer=None,
            ),
        )
        self.ha = Conv2dNormActivation(
            output_channel,
            output_channel,
            kernel_size=(horizontal_kernel, 1),
            groups=output_channel,
            activation_layer=None,
        )
        self.va = Conv2dNormActivation(
            output_channel,
            output_channel,
            kernel_size=(1, vertical_kernel),
            groups=output_channel,
            activation_layer=None,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        y = self.blocks(x)
        y = self.ha(y)
        y = self.va(y)
        y = self.activation(y)
        return F.interpolate(y, size=(h, w))


class GhostBottleNeckV2(nn.Module):

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
        horizontal_kernel: int = 3,
        vertical_kernel: int = 3,
    ):
        """GhostNet bottleneck [2211.12905v1]

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
            horizontal_kernel (int, optional): horizontal attention kernel size. Defaults to 3.
            vertical_kernel (int, optional): vertical attention kernel size. Defaults to 3.
        """
        super().__init__()

        self.conv = nn.Sequential(
            GhostModule(
                input_channel,
                expansion_size,
                s=s,
                d=d,
            ),
            nn.BatchNorm2d(expansion_size),
            nn.ReLU(inplace=True),
        )

        self.attention = DFCAttention(
            input_channel,
            expansion_size,
            horizontal_kernel,
            vertical_kernel,
        )

        self.se = None
        if enable_se:
            self.se = SqueezeExcitation2d(
                expansion_size,
                reduction_ratio,
            )

        # depthwise conv
        self.dw = None
        if stride > 1:
            self.dw = Conv2dNormActivation(
                expansion_size,
                expansion_size,
                kernel,
                stride,
                groups=expansion_size,
                activation_layer=None,
            )

        self.head = nn.Sequential(
            GhostModule(
                expansion_size,
                output_channel,
                s=s,
                d=d,
            ),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention(x) * self.conv(x)

        if self.se is not None:
            y = self.se(y)

        if self.dw is not None:
            y = self.dw(y)

        return self.head(y)


class GhostLayerV2(nn.Module):

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
        horizontal_kernel: int = 3,
        vertical_kernel: int = 3,
    ):
        """GhostNet layer [2211.12905v1]

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
            horizontal_kernel (int, optional): horizontal attention kernel size. Defaults to 3.
            vertical_kernel (int, optional): vertical attention kernel size. Defaults to 3.
        """
        super().__init__()

        block = GhostBottleNeckV2(
            input_channel,
            output_channel,
            kernel,
            stride,
            s,
            d,
            expansion_size,
            enable_se,
            reduction_ratio,
            horizontal_kernel,
            vertical_kernel,
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
