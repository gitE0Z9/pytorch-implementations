import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models import ResBlock

from ..resnet.network import BottleNeck, ConvBlock


def build_residual_unit(
    input_channel: int,
    output_channel: int,
    stride: int = 1,
) -> ResBlock:
    """pre-activation bottleneck residual unit used throughout residual attention network

    Args:
        input_channel (int): input channel size
        output_channel (int): output channel size
        stride (int, optional): stride of block. Defaults to 1.

    Returns:
        ResBlock: residual unit
    """
    return ResBlock(
        input_channel,
        output_channel,
        BottleNeck(
            input_channel,
            output_channel // 4,
            stride=stride,
            expansion_factor=4,
            pre_activation=True,
        ),
        stride=stride,
        activation=None,
    )


class DownSample(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        r: int = 1,
    ):
        """downsample part of mask branch of residual attention module

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            r (int, optional): number of residual units between adjacent pooling layer in the mask branch, according to paper. Defaults to 1.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            *[
                build_residual_unit(
                    input_channel if i == 0 else output_channel,
                    output_channel,
                )
                for i in range(r)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpSample(nn.Module):
    def __init__(
        self,
        output_channel: int,
        r: int = 1,
    ):
        """upsample part of mask branch of residual attention module

        Args:
            output_channel (int): output channel size
            r (int, optional): number of residual units between adjacent pooling layer in the mask branch, according to paper. Defaults to 1.
        """
        super().__init__()
        self.block = nn.Sequential(
            *[build_residual_unit(output_channel, output_channel) for _ in range(r)]
        )

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        y = self.block(x)
        return F.interpolate(y, size)


class MaskBranch(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        r: int = 1,
        num_skip: int = 0,
    ):
        """mask branch of residual attention module

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            r (int, optional): number of residual units between adjacent pooling layer in the mask branch, according to paper. Defaults to 1.
            num_skip (int, optional): number of skip connections. Defaults to 0.
        """
        super().__init__()
        self.num_skip = num_skip
        self.downsample_list = nn.ModuleList(
            [
                DownSample(
                    input_channel if i == 0 else output_channel,
                    output_channel,
                    r,
                )
                for i in range(num_skip + 1)
            ]
        )
        self.upsample_list = nn.ModuleList(
            [UpSample(output_channel, r) for _ in range(num_skip + 1)]
        )
        self.skip_list = nn.ModuleList(
            [
                build_residual_unit(output_channel, output_channel)
                for _ in range(num_skip)
            ]
        )
        self.post_block = nn.Sequential(
            ConvBlock(
                output_channel,
                output_channel,
                pre_activation=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass all downsampled features
        downsample_features = []
        recover_sizes = [x.shape[2:]]
        for module in self.downsample_list:
            x = module(x)
            downsample_features.append(x)

        for downsample_feature in downsample_features[:-1]:
            recover_sizes.append(downsample_feature.shape[2:])

        # the smallest feature in the top element
        downsample_features.reverse()

        # concat skip and upsample step by step
        y = downsample_features[0]
        recover_size = recover_sizes.pop()
        if self.num_skip > 0:
            # upsample_list & skip_list have no direction
            y = self.upsample_list[0](y, recover_size)
            for downsample_feature, upsample_module, skip_module in zip(
                downsample_features[1:],
                self.upsample_list[1:],
                self.skip_list,
            ):
                y += skip_module(downsample_feature)
                y = upsample_module(y, recover_sizes.pop())
        else:
            y = self.upsample_list[0](y, recover_size)

        # final conv
        y = self.post_block(y)
        return y


class AttentionModule(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        p: int = 1,
        t: int = 1,
        r: int = 1,
        num_skip: int = 0,
    ):
        """attention module in residual attention network [1704.06904v1]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            p (int, optional): number of residual unit before and after attention structure. Defaults to 1.
            t (int, optional): number of residual unit in trunk branch. Defaults to 1.
            r (int, optional): number of residual units between adjacent pooling layer in the mask branch, according to paper. Defaults to 1.
            num_skip (int, optional): number of skip connections. Defaults to 0.
        """
        super().__init__()

        self.pre_res_block = nn.Sequential(
            *[
                build_residual_unit(
                    input_channel if i == 0 else output_channel,
                    output_channel,
                )
                for i in range(p)
            ]
        )
        self.trunk_branch = nn.Sequential(
            *[build_residual_unit(output_channel, output_channel) for _ in range(t)]
        )
        self.mask_branch = MaskBranch(
            output_channel,
            output_channel,
            r,
            num_skip,
        )
        self.post_res_block = nn.Sequential(
            *[build_residual_unit(output_channel, output_channel) for _ in range(p)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_res_block(x)
        trunked_feature = self.trunk_branch(y)
        masked_feature = self.mask_branch(y)

        y = trunked_feature * (1 + masked_feature.sigmoid())
        return self.post_res_block(y)
