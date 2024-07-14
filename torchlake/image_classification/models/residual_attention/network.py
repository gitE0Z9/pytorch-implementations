import torch
import torch.nn.functional as F
from torch import nn

from ..resnet.network import BottleNeck, ConvBlock, ResBlock


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
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            *[
                ResBlock(
                    input_channel if i == 0 else output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
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
        super(UpSample, self).__init__()
        self.block = nn.Sequential(
            *[
                ResBlock(
                    output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
                )
            ]
            * r,
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
        super(MaskBranch, self).__init__()
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
            [
                UpSample(
                    output_channel,
                    r,
                )
            ]
            * (num_skip + 1)
        )
        self.skip_list = nn.ModuleList(
            [
                ResBlock(
                    output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
                )
            ]
            * num_skip
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
        """residual block in resnet
        skip connection has kernel size 1 and input_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            block (BottleNeck): block class
            p (int, optional): number of residual unit before and after attention structure. Defaults to 1.
            t (int, optional): number of residual unit in trunk branch. Defaults to 1.
            r (int, optional): number of residual units between adjacent pooling layer in the mask branch, according to paper. Defaults to 1.
            num_skip (int, optional): number of skip connections. Defaults to 0.
        """
        super(AttentionModule, self).__init__()

        self.pre_res_block = nn.Sequential(
            *[
                ResBlock(
                    input_channel if i == 0 else output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
                )
                for i in range(p)
            ]
        )
        self.trunk_branch = nn.Sequential(
            *[
                ResBlock(
                    output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
                )
            ]
            * t
        )
        self.mask_branch = MaskBranch(
            output_channel,
            output_channel,
            r,
            num_skip,
        )
        self.post_res_block = nn.Sequential(
            *[
                ResBlock(
                    output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
                )
            ]
            * p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_res_block(x)
        trunked_feature = self.trunk_branch(y)
        masked_feature = self.mask_branch(y)

        y = trunked_feature * (1 + masked_feature.sigmoid())
        return self.post_res_block(y)
