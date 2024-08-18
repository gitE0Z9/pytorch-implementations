from typing import Literal

import torch
from torch import nn
from torchlake.common.models import FlattenFeature, ResBlock
from torchvision.ops import Conv2dNormActivation

from .network import BottleNeck


class SqueezeNeXt(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 23,
        width_multiplier: float = 1,
        version: Literal[1] | Literal[2] | Literal[3] | Literal[4] | Literal[5] = 1,
    ):
        """SqueezeNeXt [1803.10615v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            num_layer (int, optional): number of layers. Defaults to 23.
            width_multiplier (float, optional): width multiplier. Defaults to 1.
            version (Literal[1] | Literal[2] | Literal[3] | Literal[4] | Literal[5] , optional): v1-v5. Defaults to 1.
        """
        super(SqueezeNeXt, self).__init__()
        # for identity check
        self.num_layer = num_layer
        self.version = version
        self.width_multiplier = width_multiplier

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                int(width_multiplier * 64),
                7 if version == 1 else 5,
                stride=2,
                padding=0,
                norm_layer=None,
            ),
            nn.MaxPool2d(3, 2),
            # middle layers
            *self.build_middle_layers(),
            # final layers
            Conv2dNormActivation(
                int(width_multiplier * 256),
                int(width_multiplier * 128),
                1,
                norm_layer=None,
            ),
        )
        self.pool = FlattenFeature()
        self.fc = nn.Linear(int(width_multiplier * 128), output_size)

    def build_middle_layers(self) -> list[nn.Module]:
        """Build middle layers."""
        # each stage channel width, (input_channel, output_channel)
        width_config = ((64, 32), (32, 64), (64, 128), (128, 256))
        # version -> list of #repeat
        block_configs = {
            1: [6, 6, 8, 1],
            2: [6, 6, 8, 1],
            3: [4, 8, 8, 1],
            4: [2, 10, 8, 1],
            5: [2, 4, 14, 1],
        }
        num_blocks = block_configs[self.version]

        middle_layers = []
        for block_index, ((in_c, out_c), num_block) in enumerate(
            zip(width_config, num_blocks)
        ):

            for layer_index in range(num_block):
                _in_c = int(
                    self.width_multiplier * (in_c if layer_index == 0 else out_c)
                )
                _out_c = int(self.width_multiplier * out_c)
                _stride = 2 if layer_index == 0 else 1
                middle_layers.append(
                    ResBlock(
                        _in_c,
                        _out_c,
                        BottleNeck(_in_c, _out_c, stride=_stride),
                        stride=_stride,
                    )
                )

        return middle_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)
