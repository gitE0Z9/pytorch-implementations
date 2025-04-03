from copy import deepcopy
import torch
from torch import nn
from torchlake.common.models import (
    DepthwiseSeparableConv2d,
    ResBlock,
    SqueezeExcitation2d,
)
from torchvision.ops import Conv2dNormActivation
from torchlake.common.helpers.fuser import (
    fuse_conv_bn,
    fuse_sum_parallel_convs,
    empty_conv,
    convert_bn_to_conv,
)

from ..ghostnet.network import GhostModule
from ..ghostnetv2.network import (
    DFCAttention,
    GhostBottleNeckV2,
    GhostLayerV2,
)


class InceptionModule(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
        num_branch: int = 1,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = kernel
        self.stride = stride
        self.groups = groups

        self.convs = nn.ModuleList(
            [
                Conv2dNormActivation(
                    input_channel,
                    output_channel,
                    kernel,
                    stride,
                    groups=groups,
                    activation_layer=None,
                )
            ]
            * num_branch
        )

        self.skip_branch = None
        if stride == 1 and input_channel == output_channel:
            self.skip_branch = nn.BatchNorm2d(output_channel)

        self.scale_branch = None
        if kernel > 1:
            self.scale_branch = Conv2dNormActivation(
                input_channel,
                output_channel,
                1,
                stride,
                groups=groups,
                activation_layer=None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sum(conv(x) for conv in self.convs)

        if self.skip_branch is not None:
            y += self.skip_branch(x)

        if self.scale_branch is not None:
            y += self.scale_branch(x)

        return y

    def reparameterize(self) -> nn.Conv2d:
        device = self.convs[0][0].weight.data.device
        dest = nn.Conv2d(
            self.input_channel,
            self.output_channel,
            self.kernel,
            self.stride,
            padding=self.kernel // 2,
            groups=self.groups,
            device=device,
        )
        # reset to zero for final sum operation
        empty_conv(dest)

        placeholder = []
        for conv in self.convs:
            temp = deepcopy(dest)
            fuse_conv_bn(conv[0], conv[1], temp)
            placeholder.append(temp)

        # merge other two branchs
        if self.scale_branch is not None:
            # since dest might have kernel larger than 1 or no bias
            temp = nn.Conv2d(
                self.input_channel,
                self.output_channel,
                1,
                self.stride,
                groups=self.groups,
                bias=True,
                device=device,
            )
            fuse_conv_bn(self.scale_branch[0], self.scale_branch[1], temp)
            placeholder.append(temp)

        if self.skip_branch is not None:
            temp = deepcopy(dest)
            convert_bn_to_conv(self.skip_branch, temp)
            placeholder.append(temp)

        fuse_sum_parallel_convs(*placeholder, dest=dest)

        return dest


class GhostModuleV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        s: int = 2,
        d: int = 1,
    ):
        """GhostNet ghost module v3 [2404.11202v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            s (int): ratio of ghost features. Defaults to 2.
            d (int): kernel size of ghost's depthwise convolution layer. Defaults to 1.
        """
        super().__init__()
        self.s = s
        self.identity_channel = output_channel // s
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
            d,
            groups=self.identity_channel,
            num_branch=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c // s, h, w
        y = self.pointwise_conv(x)

        # b, c / s + c * (s-1) / s, h, w
        return torch.cat([y, self.ghost(y)], 1)

    def reparameterize(self, dest: GhostModule):
        dest.pointwise_conv = self.pointwise_conv.reparameterize()
        dest.ghost = self.ghost.reparameterize()


class GhostBottleNeckV3(nn.Module):

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
        """GhostNet bottleneck v3 [2404.11202v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            s (int): ratio of ghost features. Defaults to 2.
            d (int): kernel size of ghost's depthwise convolution layer. Defaults to 1.
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
            cloned = deepcopy(conv)
            fuse_conv_bn(conv, self.dw[1], cloned)
            dest.dw = cloned


class GhostLayerV3(nn.Module):

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
        """GhostNet layer v3 [2404.11202v1]

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

        block = GhostBottleNeckV3(
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

    def reparameterize(self, dest: GhostLayerV2):
        self.block.block.reparameterize(dest.block.block)
        dest.block.downsample.load_state_dict(self.block.downsample.state_dict())
