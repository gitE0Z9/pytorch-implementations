from copy import deepcopy

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.helpers.fuser import (
    convert_bn_to_conv,
    empty_conv,
    fuse_conv_bn,
    fuse_sum_parallel_convs,
)
from torchlake.common.models import SqueezeExcitation2d


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
        """re-parameterizable convolution of RepVGG [2101.03697]
        num_branch conv-bn branches (more than 1 from MobileOne [2206.04040]), a 1x1 conv-bn scale branch if kernel > 1,
        and a bn skip branch if shape is unchanged, summed together.
        all branches can be fused into a single conv at inference.

        Args:
            input_channel (int): input channel size.
            output_channel (int): output channel size.
            kernel (int): kernel size.
            stride (int, optional): stride. Defaults to 1.
            groups (int, optional): number of groups. Defaults to 1.
            num_branch (int, optional): number of conv-bn branches. Defaults to 1.
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = kernel
        self.stride = stride
        self.groups = groups

        self.branches = nn.ModuleList(
            [
                Conv2dNormActivation(
                    input_channel,
                    output_channel,
                    kernel,
                    stride,
                    groups=groups,
                    activation_layer=None,
                )
                for _ in range(num_branch)
            ]
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
        y = sum(branch(x) for branch in self.branches)

        if self.skip_branch is not None:
            y += self.skip_branch(x)

        if self.scale_branch is not None:
            y += self.scale_branch(x)

        return y

    def reparameterize(self) -> nn.Conv2d:
        device = self.branches[0][0].weight.data.device
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
        for conv in self.branches:
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


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
        num_branch: int = 1,
        enable_se: bool = False,
        reduction_ratio: float = 16,
    ):
        """RepVGG block [2101.03697]
        re-parameterizable convolution -> (squeeze excitation) -> relu
        RepVGG uses a single branch, MobileOne [2206.04040] uses multiple branches.

        Args:
            input_channel (int): input channel size.
            output_channel (int): output channel size.
            kernel (int): kernel size.
            stride (int, optional): stride. Defaults to 1.
            groups (int, optional): number of groups. Defaults to 1.
            num_branch (int, optional): number of conv-bn branches. Defaults to 1.
            enable_se (bool, optional): enable squeeze excitation. Defaults to False.
            reduction_ratio (float, optional): reduction ratio. Defaults to 16.
        """
        super().__init__()
        self.layers = nn.Sequential(
            InceptionModule(
                input_channel,
                output_channel,
                kernel,
                stride,
                groups=groups,
                num_branch=num_branch,
            ),
            nn.ReLU(True),
        )

        if enable_se:
            self.layers.insert(-1, SqueezeExcitation2d(output_channel, reduction_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
