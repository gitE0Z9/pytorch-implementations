import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase

from .network import ConvBlock, DownSampling, UpSampling


class UNet(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        hidden_dim: int = 64,
    ):
        self.hidden_dim = hidden_dim
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            ConvBlock(input_channel, self.hidden_dim),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                DownSampling(self.hidden_dim * (2**i), self.hidden_dim * (2 ** (i + 1)))
                for i in range(4)
            ]
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                UpSampling(self.hidden_dim * (2**i), self.hidden_dim * (2 ** (i - 1)))
                for i in range(4, 0, -1)
            ]
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)

        features = [y]
        for block in self.blocks[:-1]:
            y = block(y)
            features.append(y)

        y = self.blocks[-1](y)

        for neck in self.neck:
            y = neck(y, features.pop())

        y = self.head(y)
        return y
