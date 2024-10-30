from typing import Any
from torch import nn

from ..ghostnet.model import GhostNet
from .network import GhostLayerV2


class GhostNetV2(GhostNet):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """GhostNet version 2 [2211.12905v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        self.width_multiplier = width_multiplier
        super().__init__(input_channel, output_size)

    @property
    def config(self) -> list[list[Any]]:
        return [
            # input_channel, output_channel, kernel, stride, expansion_size, enable_se
            (16, 16, 3, 1, 16, False),
            (16, 24, 3, 2, 48, False),
            (24, 24, 3, 1, 72, False),
            (24, 40, 5, 2, 72, True),
            (40, 40, 5, 1, 120, True),
            (40, 80, 3, 2, 240, False),  # kernel: 5 -> 3 in following layers
            (80, 80, 3, 1, 200, False),
            (80, 80, 3, 1, 184, False),
            (80, 80, 3, 1, 184, False),
            (80, 112, 3, 1, 480, True),
            (112, 112, 3, 1, 672, True),
            (112, 160, 5, 2, 672, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
        ]

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                GhostLayerV2(
                    int(in_c * self.width_multiplier),
                    int(out_c * self.width_multiplier),
                    kernel,
                    stride=stride,
                    s=2,
                    d=3,
                    expansion_size=expansion_size,
                    enable_se=enable_se,
                    horizontal_kernel=5,
                    vertical_kernel=5,
                )
                for in_c, out_c, kernel, stride, expansion_size, enable_se in self.config
            ]
        )
