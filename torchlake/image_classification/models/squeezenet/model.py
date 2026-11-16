from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import FlattenFeature, ResBlock
from torchlake.common.models.model_base import ModelBase

from .network import FireModule


class SqueezeNet(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.5,
        version: Literal["1.0", "1.1"] = "1.0",
    ):
        """SqueezeNet [1602.07360v4]
        1.1: https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            version (Literal["1.0", "1.1"] , optional): version 1.0 and 1.1. Defaults to "1.0".
        """
        self.dropout_prob = dropout_prob
        self.version = version
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 512

    @property
    def entry_channel(self) -> int:
        return 96 if self.version == "1.0" else 64

    @property
    def config(self) -> tuple[tuple[int | float]]:
        # input_channel, squeeze_ratio, expand_ratio of fire 2 ~ fire 9
        return (
            # fire 2
            (self.entry_channel, 16 / self.entry_channel, 4),
            (128, 1 / 8, 4),
            # 1.1 pool here
            (128, 1 / 4, 4),
            # 1.0 pool here
            (256, 1 / 8, 4),
            # 1.1 pool here
            (256, 3 / 16, 4),
            (384, 1 / 8, 4),
            (384, 1 / 6, 4),
            # 1.0 pool here
            # fire 9
            (512, 1 / 8, 4),
        )

    @property
    def pool_indices(self) -> tuple[int]:
        """index of fire module followed by max pooling"""
        return {
            "1.0": (2, 6),  # after fire 4, 8
            "1.1": (1, 3),  # after fire 3, 5
        }[self.version]

    @property
    def residual_indices(self) -> tuple[int]:
        """index of fire module with simple bypass, i.e. fire 3, 5, 7, 9"""
        return (1, 3, 5, 7)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.entry_channel,
                7 if self.version == "1.0" else 3,
                stride=2,
                padding=2,  # as paper
                norm_layer=None,
            ),
            # padding as paper
            nn.MaxPool2d(3, 2, padding=0),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for i, (in_c, squeeze_ratio, expand_ratio) in enumerate(self.config):
            layer = FireModule(in_c, squeeze_ratio, expand_ratio)
            # simple bypass
            if i in self.residual_indices:
                layer = ResBlock(
                    in_c,
                    int(in_c * squeeze_ratio * expand_ratio * 2),
                    layer,
                    activation=None,
                )
            self.blocks.append(layer)

            if i in self.pool_indices:
                # padding as paper
                self.blocks.append(nn.MaxPool2d(3, 2, padding=0))

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            Conv2dNormActivation(
                self.feature_dim,
                output_size,
                1,
                norm_layer=None,
            ),
            FlattenFeature(),
        )
