import torch
from torch import nn
from torchlake.common.network import ConvBnRelu
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
            ConvBnRelu(input_channel, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2),
        )
        self.build_blocks(num_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(CONFIGS[num_layer][-1][1], output_size)

    def build_blocks(self, num_layer: int):
        config = CONFIGS[num_layer]

        for block_index, (
            input_channel,
            output_channel,
            base_number,
            num_layer,
            layer_class,
        ) in enumerate(config):
            layers = [
                ResBlock(
                    input_channel if layer_index == 0 else output_channel,
                    base_number,
                    output_channel,
                    layer_class,
                )
                for layer_index in range(num_layer)
            ]
            if block_index not in [0, len(config) - 1]:
                layers.append(nn.MaxPool2d(2, 2))

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        for i in range(4):
            block = getattr(self, f"block{i+1}", None)
            if not block:
                break
            y = block(y)
        y = self.pool(y)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        return y
