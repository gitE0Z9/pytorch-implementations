import torch
from torch import nn

from .network import BottleNeck, ConvBlock, ResBlock

# input, output, base?, number_block, block_type
CONFIGS = {
    18: [
        [64, 64, 64, 2, ConvBlock],  # less block
        [64, 128, 128, 2, ConvBlock],  # less block
        [128, 256, 256, 2, ConvBlock],  # less block
        [256, 512, 512, 2, ConvBlock],  # less block
    ],
    34: [
        [64, 64, 64, 3, ConvBlock],  # narrower
        [64, 128, 128, 4, ConvBlock],  # narrower
        [128, 256, 256, 6, ConvBlock],  # narrower
        [256, 512, 512, 3, ConvBlock],  # narrower
    ],
    50: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 4, BottleNeck],
        [512, 1024, 256, 6, BottleNeck],
        [1024, 2048, 512, 3, BottleNeck],
    ],
    101: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 4, BottleNeck],
        [512, 1024, 256, 23, BottleNeck],  # more block
        [1024, 2048, 512, 3, BottleNeck],
    ],
    152: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 8, BottleNeck],  # more block
        [512, 1024, 256, 36, BottleNeck],  # more block
        [1024, 2048, 512, 3, BottleNeck],
    ],
}


class ResNet(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 50,
    ):
        super(ResNet, self).__init__()
        self.foot = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2),
        )
        self.build_blocks(num_layer)
        self.fc = nn.Linear(CONFIGS[num_layer][-1][1], output_size)

    def build_blocks(self, num_layer: int):
        config = CONFIGS[num_layer]

        self.block1 = nn.Sequential(
            *[
                ResBlock(
                    config[0][0] if i == 0 else config[0][1],
                    config[0][2],
                    config[0][1],
                    config[0][-1],
                )
                for i in range(config[0][-2])
            ]
        )
        self.block2 = nn.Sequential(
            *[
                ResBlock(
                    config[1][0] if i == 0 else config[1][1],
                    config[1][2],
                    config[1][1],
                    config[1][-1],
                )
                for i in range(config[1][-2])
            ],
            nn.MaxPool2d(2, 2),
        )

        self.block3 = nn.Sequential(
            *[
                ResBlock(
                    config[2][0] if i == 0 else config[2][1],
                    config[2][2],
                    config[2][1],
                    config[2][-1],
                )
                for i in range(config[2][-2])
            ],
            nn.MaxPool2d(2, 2),
        )

        self.block4 = nn.Sequential(
            *[
                ResBlock(
                    config[3][0] if i == 0 else config[3][1],
                    config[3][2],
                    config[3][1],
                    config[3][-1],
                )
                for i in range(config[3][-2])
            ],
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        for i in range(4):
            y = getattr(self, f"block{i+1}")(y)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        return y
