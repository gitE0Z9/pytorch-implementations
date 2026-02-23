import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase

from .network import BottleNeck, Stem, UpsamplingBlock


class ENet(ModelBase):
    def __init__(self, input_channel: int, output_size: int):
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 16

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Stem(input_channel, 13),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                # 1.0 - 1.x
                BottleNeck(16, 64, stride=2, dropout_prob=0.01),
                BottleNeck(64, 64, dropout_prob=0.01),
                BottleNeck(64, 64, dropout_prob=0.01),
                BottleNeck(64, 64, dropout_prob=0.01),
                BottleNeck(64, 64, dropout_prob=0.01),
                # 2.0 - 2.8
                BottleNeck(64, 128, stride=2),
                BottleNeck(128, 128),
                BottleNeck(128, 128, dilation=2),
                BottleNeck(128, 128, asymmetric=True),
                BottleNeck(128, 128, dilation=4),
                BottleNeck(128, 128),
                BottleNeck(128, 128, dilation=8),
                BottleNeck(128, 128, asymmetric=True),
                BottleNeck(128, 128, dilation=16),
                # 3.1 - 3.8
                BottleNeck(128, 128),
                BottleNeck(128, 128, dilation=2),
                BottleNeck(128, 128, asymmetric=True),
                BottleNeck(128, 128, dilation=4),
                BottleNeck(128, 128),
                BottleNeck(128, 128, dilation=8),
                BottleNeck(128, 128, asymmetric=True),
                BottleNeck(128, 128, dilation=16),
            ]
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                # 4.0 - 4.2
                UpsamplingBlock(128, 64),
                BottleNeck(64, 64),
                BottleNeck(64, 64),
                # 5.0 - 5.1
                UpsamplingBlock(64, 16),
                BottleNeck(16, 16),
            ]
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, output_size, 2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)

        indices = []
        for block in self.blocks:
            y = block(y)
            if block.stride > 1:
                y, index = y
                indices.append(index)

        for neck in self.neck:
            if isinstance(neck, UpsamplingBlock):
                y = neck(y, indices.pop())
            else:
                y = neck(y)

        return self.head(y)
