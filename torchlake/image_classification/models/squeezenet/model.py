from typing import Literal

import torch
from torch import nn
from torchlake.common.models import FlattenFeature, ResBlock
from torchvision.ops import Conv2dNormActivation

from .network import FireModule


class SqueezeNet(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.5,
        version: Literal["1.0"] | Literal["1.1"] = "1.0",
    ):
        """SqueezeNet [1602.07360v4]
        1.1: https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            version (Literal["1.0"] | Literal["1.1"] , optional): version 1.0 and 1.1. Defaults to "1.0".
        """
        super(SqueezeNet, self).__init__()
        self.version = version

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                96 if version == "1.0" else 64,
                7 if version == "1.0" else 3,
                stride=2,
                norm_layer=None,
            ),
            nn.MaxPool2d(3, 2, 1),
            # middle layers
            *self.build_middle_layers(),
            # final layers
            nn.Dropout(dropout_prob),
            Conv2dNormActivation(
                512,
                output_size,
                1,
                norm_layer=None,
            ),
        )
        self.pool = FlattenFeature()

    def build_middle_layers(self) -> list[nn.Module]:
        """Build middle layers."""
        entry_channel_size = 96 if self.version == "1.0" else 64
        # input_channel, squeeze_ratio, expand_ratio
        config = [
            # fire 2
            (entry_channel_size, 16 / entry_channel_size, 4),
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
        ]
        pool_configs = {
            "1.0": [2, 6],  # 1, 4, 8
            "1.1": [1, 3],  # 1, 3, 5
        }
        pool_config = pool_configs[self.version]
        residual_config = [1, 3, 5, 7]  # fire 3, 5, 7, 9

        middle_layers = []
        for i, (in_c, squeeze_ratio, expand_ratio) in enumerate(config):

            # common block
            layer = FireModule(in_c, squeeze_ratio, expand_ratio)
            # if simple bypass
            if i in residual_config:
                layer = ResBlock(
                    in_c,
                    int(in_c * squeeze_ratio * expand_ratio * 2),
                    layer,
                    activation=None,
                )
            middle_layers.append(layer)

            # pool
            if i in pool_config:
                middle_layers.append(nn.MaxPool2d(3, 2, 1))

        return middle_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return self.pool(y)
