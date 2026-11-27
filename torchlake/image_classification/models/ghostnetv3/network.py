from copy import deepcopy

import torch
from torch import nn

from torchlake.common.helpers.fuser import fuse_conv_bn
from torchlake.common.models import (
    DepthwiseSeparableConv2d,
    ResBlock,
    SqueezeExcitation2d,
)

from ..ghostnet.network import GhostModule
from ..ghostnetv2.network import DFCAttention, GhostBlockV2, GhostBottleNeckV2
from ..repvgg.network import InceptionModule


class GhostModuleV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        num_ghost_feature: int = 2,
        kernel_ds: int = 1,
    ):
        """GhostNet ghost module v3 [2404.11202v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            num_ghost_feature (int): ratio of ghost features. Defaults to 2.
            kernel_ds (int): kernel size of ghost's depthwise convolution layer. Defaults to 1.
        """
        super().__init__()
        self.num_ghost_feature = num_ghost_feature
        self.identity_channel = output_channel // num_ghost_feature
        transformed_channel = output_channel - self.identity_channel

        self.pointwise_conv = InceptionModule(
            input_channel,
            self.identity_channel,
            1,
            num_branch=3,
        )
        self.ghost = InceptionModule(
            self.identity_channel,
            transformed_channel,
            kernel_ds,
            groups=self.identity_channel,
            num_branch=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c // num_ghost_feature, h, w
        y = self.pointwise_conv(x)

        # b, c / num_ghost_feature + c * (num_ghost_feature - 1) / num_ghost_feature, h, w
        return torch.cat([y, self.ghost(y)], 1)

    def reparameterize(self, dest: GhostModule):
        # load into existing modules to keep state dict keys of GhostModule
        dest.pointwise_conv.load_state_dict(
            self.pointwise_conv.reparameterize().state_dict()
        )
        dest.ghost.load_state_dict(self.ghost.reparameterize().state_dict())


class GhostBottleNeckV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        num_ghost_feature: int = 2,
        kernel_ds: int = 1,
        expansion_size: int = 1,
        enable_se: bool = True,
        reduction_ratio: int = 4,
        horizontal_kernel: int = 3,
        vertical_kernel: int = 3,
    ):
        """GhostNet bottleneck v3 [2404.11202v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            num_ghost_feature (int): ratio of ghost features. Defaults to 2.
            kernel_ds (int): kernel size of ghost's depthwise convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
            horizontal_kernel (int, optional): horizontal attention kernel size. Defaults to 3.
            vertical_kernel (int, optional): vertical attention kernel size. Defaults to 3.
        """
        super().__init__()

        self.conv = nn.Sequential(
            GhostModuleV3(
                input_channel,
                expansion_size,
                num_ghost_feature=num_ghost_feature,
                kernel_ds=kernel_ds,
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
                # activations=(nn.ReLU(True), nn.Hardsigmoid()),
            )

        # depthwise conv
        self.dw = None
        if stride > 1:
            self.dw = nn.Sequential(
                InceptionModule(
                    expansion_size,
                    expansion_size,
                    kernel,
                    stride,
                    groups=expansion_size,
                ),
                nn.BatchNorm2d(expansion_size),
            )

        self.head = nn.Sequential(
            GhostModuleV3(
                expansion_size,
                output_channel,
                num_ghost_feature=num_ghost_feature,
                kernel_ds=kernel_ds,
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

    def reparameterize(self, dest: GhostBottleNeckV2):
        # conv
        self.conv[0].reparameterize(dest.conv[0])
        dest.conv[1].load_state_dict(self.conv[1].state_dict())

        # dfc attention
        dest.attention.load_state_dict(self.attention.state_dict())

        # se
        if self.se is not None:
            dest.se.load_state_dict(self.se.state_dict())

        # head
        self.head[0].reparameterize(dest.head[0])
        dest.head[1].load_state_dict(self.head[1].state_dict())

        # dw
        if self.dw is not None:
            conv = self.dw[0].reparameterize()
            fused = deepcopy(conv)
            fuse_conv_bn(conv, self.dw[1], fused)

            # don't bother bn in dest.dw, since layer fusion in pytorch won't bring much benefits
            # should leave that to the inference engine
            dest_conv, dest_bn = dest.dw[0], dest.dw[1]
            with torch.no_grad():
                dest_conv.weight.copy_(fused.weight)
                dest_bn.weight.fill_(1)
                dest_bn.bias.copy_(fused.bias)
                dest_bn.running_mean.zero_()
                dest_bn.running_var.fill_(1 - dest_bn.eps)


class GhostBlockV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        num_ghost_feature: int = 2,
        kernel_ds: int = 1,
        expansion_size: int = 1,
        enable_se: bool = True,
        reduction_ratio: int = 4,
        horizontal_kernel: int = 3,
        vertical_kernel: int = 3,
    ):
        """GhostNet block v3 [2404.11202v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            num_ghost_feature (int): num of ghost features, group parameter. Defaults to 2.
            kernel_ds (int): kernel size of depthwise convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            reduction_ratio (int, optional): reduction ratio. Defaults to 4.
            horizontal_kernel (int, optional): horizontal attention kernel size. Defaults to 3.
            vertical_kernel (int, optional): vertical attention kernel size. Defaults to 3.
        """
        super().__init__()

        block = GhostBottleNeckV3(
            input_channel,
            output_channel,
            kernel,
            stride,
            num_ghost_feature,
            kernel_ds,
            expansion_size,
            enable_se,
            reduction_ratio,
            horizontal_kernel,
            vertical_kernel,
        )

        shortcut = None
        if input_channel != output_channel or stride > 1:
            shortcut = DepthwiseSeparableConv2d(
                input_channel,
                output_channel,
                kernel,
                stride,
                padding=kernel // 2,
                activations=(None, None),
            )

        # packed into resblock
        self.block = ResBlock(
            input_channel,  # omit
            output_channel,  # omit
            block,
            stride,  # omit
            activation=None,  # omit
            shortcut=shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def reparameterize(self, dest: GhostBlockV2):
        self.block.block.reparameterize(dest.block.block)
        dest.block.shortcut.load_state_dict(self.block.shortcut.state_dict())
