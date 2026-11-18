from typing import Any

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock

from .network import GhostBottleNeck


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
            # stage 1
            (16, 16, 3, 1, 16, False),
            (16, 24, 3, 2, 48, False),
            # stage 2
            (24, 24, 3, 1, 72, False),
            (24, 40, 5, 2, 72, True),
            # stage 3
            (40, 40, 5, 1, 120, True),
            (40, 80, 3, 2, 240, False),
            # stage 4
            (80, 80, 3, 1, 200, False),
            (80, 80, 3, 1, 184, False),
            (80, 80, 3, 1, 184, False),
            (80, 112, 3, 1, 480, True),
            (112, 112, 3, 1, 672, True),
            (112, 160, 5, 2, 672, True),
            # stage 5
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
        ]

    def build_foot(self, input_channel):
        self.foot = Conv2dNormActivation(
            input_channel,
            int(self.config[0][0] * self.width_multiplier),
            stride=2,
        )

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                ResBlock(
                    int(in_c * self.width_multiplier),
                    int(out_c * self.width_multiplier),
                    block=GhostBottleNeck(
                        int(in_c * self.width_multiplier),
                        int(out_c * self.width_multiplier),
                        kernel,
                        stride=stride,
                        num_ghost_feature=2,
                        kernel_ds=3,
                        expansion_size=int(expansion_size * self.width_multiplier),
                        enable_se=enable_se,
                    ),
                    stride=stride,
                    activation=None,
                )
                for in_c, out_c, kernel, stride, expansion_size, enable_se in self.config
            ]
        )

    def build_neck(self):
        self.neck = Conv2dNormActivation(
            int(self.config[-1][1] * self.width_multiplier),
            int(self.config[-1][4] * self.width_multiplier),
            1,
        )

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            FlattenFeature(),
            nn.Linear(
                int(self.config[-1][4] * self.width_multiplier),
                self.feature_dim,
            ),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, output_size),
        )
