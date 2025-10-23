from torch import nn
import torch

from torchlake.common.models.conv import ConvBnRelu
from torchlake.common.models.model_base import ModelBase

from .network import TransformModule


class PointNet(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.7,
    ):
        self.dropout_prob = dropout_prob
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 1024

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.ModuleDict(
            {
                "transform": TransformModule(input_channel),
                "block": nn.Sequential(
                    ConvBnRelu(input_channel, 64, 1, dimension="1d"),
                    ConvBnRelu(64, 64, 1, dimension="1d"),
                ),
            }
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleDict(
            {
                "transform": TransformModule(64),
                "block": nn.Sequential(
                    ConvBnRelu(64, 64, 1, dimension="1d"),
                    ConvBnRelu(64, 128, 1, dimension="1d"),
                    ConvBnRelu(128, 1024, 1, dimension="1d"),
                ),
            }
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d((1)),
            ConvBnRelu(self.feature_dim, 512, 1, dimension="1d"),
            nn.Dropout(p=self.dropout_prob),
            ConvBnRelu(512, 256, 1, dimension="1d"),
            nn.Dropout(p=self.dropout_prob),
            nn.Flatten(),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_affine = self.training

        # stage1
        # N, 3 => h, N
        y = x.transpose(-1, -2)
        y = self.foot["transform"](y, output_affine=output_affine)
        if output_affine:
            y, t1 = y
        y = self.foot["block"](y)

        # stage2
        # h, N => h, N
        y = self.blocks["transform"](y, output_affine=output_affine)
        if output_affine:
            y, t2 = y
        y = self.blocks["block"](y)

        # fc
        y = self.head(y)
        if output_affine:
            return y, (t1, t2)

        return y
