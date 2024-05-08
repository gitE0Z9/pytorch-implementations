from typing import Any

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import DenseBlock, TransitionBlock

# num_layer
CONFIGS = {
    121: [6, 12, 24, 16],
    169: [6, 12, 32, 32],
    201: [6, 12, 48, 32],
    264: [6, 12, 64, 48],
}

# just another resnet


class DenseNet(nn.Module):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 169,
        growth_rate: int = 32,
        compression_ratio: float = 0.5,
        configs: dict[int, Any] = CONFIGS,
    ):
        super(DenseNet, self).__init__()
        self.config = configs[num_layer]
        self.growth_rate = growth_rate
        self.compression_ratio = compression_ratio

        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 7, stride=2),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.build_blocks()

        embed_dim = (
            getattr(self, f"block{len(self.config)}", None)[0]
            .layers[-1][0]
            .conv.in_channels
            + growth_rate
        )
        self.pool = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(embed_dim, output_size)

    def build_blocks(self):
        input_channel = 64
        for block_index, num_layer in enumerate(self.config):
            layers = [
                DenseBlock(
                    input_channel,
                    num_layer,
                    growth_rate=self.growth_rate,
                )
            ]
            if block_index != len(self.config) - 1:
                output_channel = self.growth_rate * num_layer + input_channel
                compressed_channel = int(output_channel * self.compression_ratio)

                layers.append(TransitionBlock(output_channel, compressed_channel))
                input_channel = compressed_channel

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        for i, _ in enumerate(self.config):
            block = getattr(self, f"block{i+1}", None)
            if not block:
                break
            y = block(y)
        y = self.pool(y)
        y = self.fc(y)

        return y
