from typing import Any

from torch import nn
from torchvision.ops import Conv2dNormActivation

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
                    horizontal_kernel=5,
                    vertical_kernel=5,
                )
                for in_c, out_c, kernel, stride, expansion_size, _ in self.config
            ]
        )
