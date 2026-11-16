from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import ResBlock
from torchlake.common.models.model_base import ModelBase

from .network import BottleNeck


class SqueezeNeXt(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 23,
        width_multiplier: float = 1,
        version: Literal[1, 2, 3, 4, 5] = 1,
        config: tuple[tuple[int]] | None = None,
    ):
        """SqueezeNeXt [1803.10615v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            num_layer (int, optional): number of layers. Defaults to 23.
            width_multiplier (float, optional): width multiplier. Defaults to 1.
            version (Literal[1, 2, 3, 4, 5], optional): v1-v5. Defaults to 1.
            config (tuple[tuple[int]], optional): stage config, format is (input_channel, output_channel, num_block). Defaults to None.
        """
        # for identity check
        self.num_layer = num_layer
        self.width_multiplier = width_multiplier
        self.version = version
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return int(self.width_multiplier * 128)

    @property
    def config(self) -> tuple[tuple[int]]:
        if self._config:
            return self._config

        # stage width, (input_channel, output_channel)
        width_config = ((64, 32), (32, 64), (64, 128), (128, 256))
        # version -> number of blocks of each stage
        num_blocks = {
            1: (6, 6, 8, 1),
            2: (6, 6, 8, 1),
            3: (4, 8, 8, 1),
            4: (2, 10, 8, 1),
            5: (2, 4, 14, 1),
        }[self.version]

        # input_channel, output_channel, num_block
        return tuple(
            (in_c, out_c, num_block)
            for (in_c, out_c), num_block in zip(width_config, num_blocks)
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                int(self.width_multiplier * self.config[0][0]),
                7 if self.version == 1 else 5,
                stride=2,
                padding=0,
                norm_layer=None,
            ),
            nn.MaxPool2d(3, 2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for in_c, out_c, num_block in self.config:
            for block_idx in range(num_block):
                _in_c = int(self.width_multiplier * (in_c if block_idx == 0 else out_c))
                _out_c = int(self.width_multiplier * out_c)
                stride = 2 if block_idx == 0 else 1
                self.blocks.append(
                    ResBlock(
                        _in_c,
                        _out_c,
                        BottleNeck(_in_c, _out_c, stride=stride),
                        stride=stride,
                    )
                )

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(
            0,
            Conv2dNormActivation(
                int(self.width_multiplier * self.config[-1][1]),
                self.feature_dim,
                1,
                norm_layer=None,
            ),
        )
