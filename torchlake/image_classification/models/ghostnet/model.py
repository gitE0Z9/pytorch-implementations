from typing import Any
from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from .network import GhostLayer


class GhostNet(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """GhostNet version 1 [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        self.width_multiplier = width_multiplier
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return int(1280 * self.width_multiplier)

    @property
    def config(self) -> list[list[Any]]:
        return [
            # input_channel, output_channel, kernel, stride, expansion_size, enable_se
            (16, 16, 3, 1, 16, False),
            (16, 24, 3, 2, 48, False),
            (24, 24, 3, 1, 72, False),
            (24, 40, 5, 2, 72, True),
            (40, 40, 5, 1, 120, True),
            (40, 80, 5, 2, 240, False),
            (80, 80, 5, 1, 200, False),
            (80, 80, 5, 1, 184, False),
            (80, 80, 5, 1, 184, False),
            (80, 112, 5, 1, 480, True),
            (112, 112, 5, 1, 672, True),
            (112, 160, 5, 2, 672, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
        ]

    def build_foot(self, input_channel):
        self.foot = Conv2dNormActivation(
            input_channel,
            int(16 * self.width_multiplier),
            stride=2,
        )

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                GhostLayer(
                    int(in_c * self.width_multiplier),
                    int(out_c * self.width_multiplier),
                    kernel,
                    stride=stride,
                    s=2,
                    d=3,
                    expansion_size=int(expansion_size * self.width_multiplier),
                    enable_se=enable_se,
                )
                for in_c, out_c, kernel, stride, expansion_size, enable_se in self.config
            ]
        )

    def build_neck(self):
        self.neck = nn.Sequential(
            Conv2dNormActivation(
                int(160 * self.width_multiplier),
                int(960 * self.width_multiplier),
                1,
            ),
            Conv2dNormActivation(
                int(960 * self.width_multiplier),
                int(1280 * self.width_multiplier),
                1,
            ),
        )
