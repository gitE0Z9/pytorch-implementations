import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.mixins.network import SEMixin
from torchlake.common.models import SqueezeExcitation2d


class GhostModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        num_ghost_feature: int = 2,
        kernel_ds: int = 1,
    ):
        """GhostNet ghost module [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            num_ghost_feature (int): num of ghost features, split ratio of output features for depthwise convolution. Defaults to 2.
            kernel_ds (int): kernel size of depthwise convolution layer. Defaults to 1.
        """
        super().__init__()
        self.num_ghost_feature = num_ghost_feature
        self.identity_channel = output_channel // num_ghost_feature
        transformed_channel = output_channel - self.identity_channel

        self.pointwise_conv = nn.Conv2d(input_channel, self.identity_channel, 1)

        self.ghost = nn.Conv2d(
            self.identity_channel,
            transformed_channel,
            kernel_ds,
            groups=self.identity_channel,
            padding=kernel_ds // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c // num_ghost_feature, h, w
        y = self.pointwise_conv(x)

        # b, c / num_ghost_feature + c * (num_ghost_feature - 1) / num_ghost_feature, h, w
        return torch.cat([y, self.ghost(y)], 1)


class GhostBottleNeck(SEMixin, nn.Module):

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
    ):
        """GhostNet bottleneck [1911.11907v2]

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
        """
        super().__init__()

        # basic block
        self.block = self.build_block(
            input_channel,
            output_channel,
            kernel,
            stride,
            num_ghost_feature,
            kernel_ds,
            expansion_size,
        )

        self.se = None
        if enable_se:
            self.se = SqueezeExcitation2d(
                output_channel,
                reduction_ratio,
                # activations=(nn.ReLU(True), nn.Hardsigmoid()),
            )

    def build_block(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        num_ghost_feature: int = 2,
        kernel_ds: int = 1,
        expansion_size: int = 1,
    ) -> nn.Module:
        """Block in GhostNet bottleneck [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_channel (int, optional): output channel size. Defaults to 1.
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            num_ghost_feature (int): num of ghost features, group parameter. Defaults to 2.
            kernel_ds (int): kernel size of depthwise convolution layer. Defaults to 1.
            expansion_size (int, optional): expansion size. Defaults to 1.
        """
        block = [
            GhostModule(
                input_channel,
                expansion_size,
                num_ghost_feature=num_ghost_feature,
                kernel_ds=kernel_ds,
            ),
            nn.BatchNorm2d(expansion_size),
            nn.ReLU(inplace=True),
            GhostModule(
                expansion_size,
                output_channel,
                num_ghost_feature=num_ghost_feature,
                kernel_ds=kernel_ds,
            ),
            nn.BatchNorm2d(output_channel),
        ]

        if stride > 1:
            block.insert(
                -2,
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
        return self.block(x)
