# https://github.com/fwang91/residual-attention-network
from typing import Any

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..resnet.network import BottleNeck, ResBlock
from .network import AttentionModule

# input, output, base, num_attention_module, num_skip
CONFIGS = {
    56: [
        [64, 256, 64, 1, 2],
        [256, 512, 128, 1, 1],
        [512, 1024, 256, 1, 0],
        [1024, 2048, 512, 0, 0],
    ],
    92: [
        [64, 256, 64, 1, 2],
        [256, 512, 128, 2, 1],
        [512, 1024, 256, 3, 0],
        [1024, 2048, 512, 0, 0],
    ],
}


class ResidualAttentionNetwork(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 56,
        pre_activation: bool = True,
        configs: dict[int, Any] = CONFIGS,
    ):
        super(ResidualAttentionNetwork, self).__init__()
        self.pre_activation = pre_activation
        self.config = configs[num_layer]

        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 7, stride=2, norm_layer=None),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.build_blocks()
        self.block4 = nn.Sequential(
            *[
                ResBlock(
                    self.config[-1][0] if i == 0 else self.config[-1][1],
                    self.config[-1][2],
                    self.config[-1][1],
                    BottleNeck,
                    pre_activation=True,
                )
                for i in range(3)
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(self.config[-1][1], output_size)

    def build_blocks(self):
        for block_index, (
            input_channel,
            output_channel,
            _,
            num_attention_module,
            num_skip,
        ) in enumerate(self.config):
            layers: list = [
                AttentionModule(
                    input_channel if i == 0 else output_channel,
                    output_channel,
                    p=1,
                    t=2,
                    r=1,
                    num_skip=num_skip,
                )
                for i in range(num_attention_module)
            ]
            layers.append(
                ResBlock(
                    output_channel,
                    output_channel // 4,
                    output_channel,
                    BottleNeck,
                    pre_activation=True,
                ),
            )

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        for i in range(len(self.config)):
            block = getattr(self, f"block{i+1}", None)
            if not block:
                break
            y = block(y)
        y = self.pool(y)
        y = self.fc(y)

        return y
