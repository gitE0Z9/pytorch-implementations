import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

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
        pre_activation: bool = False,
        configs: dict[int, list[list[int | nn.Module]]] = CONFIGS,
    ):
        """ResNet

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            num_layer (int, optional): number of layers. Defaults to 50.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            configs (dict[int, list[list[int | nn.Module]]], optional): configs for resnet, key is number of layers. Defaults to CONFIGS.
        """
        super(ResNet, self).__init__()
        self.pre_activation = pre_activation
        self.config = configs[num_layer]

        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, self.config[0][0], 7, stride=2),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.build_blocks()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(self.config[-1][1], output_size)

    def build_blocks(self):
        for block_index, (
            input_channel,
            output_channel,
            base_number,
            num_layer,
            layer_class,
        ) in enumerate(self.config):
            layers = [
                ResBlock(
                    input_channel if layer_index == 0 else output_channel,
                    base_number,
                    output_channel,
                    layer_class,
                    stride=2 if layer_index == 0 else 1,
                    pre_activation=self.pre_activation,
                )
                for layer_index in range(num_layer)
            ]

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        for i in range(len(self.config)):
            block = getattr(self, f"block{i+1}", None)
            y = block(y)
        y = self.pool(y)
        y = self.fc(y)

        return y
