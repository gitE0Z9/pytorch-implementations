import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import ResBlock

# input, output, number_block
CONFIGS = {
    1: [
        [24, 144, 4],
        [144, 288, 8],
        [288, 576, 4],
    ],
    2: [
        [24, 200, 4],
        [200, 400, 8],
        [400, 800, 4],
    ],
    3: [
        [24, 240, 4],
        [240, 480, 8],
        [480, 960, 4],
    ],
    4: [
        [24, 272, 4],
        [272, 544, 8],
        [544, 1088, 4],
    ],
    8: [
        [24, 384, 4],
        [384, 768, 8],
        [768, 1536, 4],
    ],
}


class ShuffleNet(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        groups: int = 1,
        scale_factor: float = 1,
        configs: dict[int, list[list[int]]] = CONFIGS,
    ):
        """ShuffleNet

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            groups (int, optional): number of groups of channels. Defaults to 1.
            scale_factor (float, optional): scale factor s. Defaults to 1.
            configs (list[list[int]], optional): configs for shufflenet, key is number of groups. Defaults to CONFIGS.
        """
        super(ShuffleNet, self).__init__()
        self.config = configs[groups]
        self.num_stage = len(self.config)
        # for identity check
        self.scale_factor = scale_factor

        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                int(scale_factor * self.config[0][0]),
                3,
                stride=2,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.build_blocks(groups, scale_factor)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(
            int(scale_factor * self.config[-1][1]),
            output_size,
        )

    def build_blocks(self, groups: int = 1, scale_factor: float = 1):
        """build blocks

        Args:
            groups (int, optional): number of groups of channels. Defaults to 1.
            scale_factor (float, optional): scale factor s. Defaults to 1.
        """
        for block_index, (
            input_channel,
            output_channel,
            num_layer,
        ) in enumerate(self.config):
            _input_channel = int(scale_factor * input_channel)
            _output_channel = int(scale_factor * output_channel)

            layers = [
                ResBlock(
                    _input_channel if layer_index == 0 else _output_channel,
                    _output_channel,
                    stride=2 if layer_index == 0 else 1,
                    groups=groups,
                )
                for layer_index in range(num_layer)
            ]

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        for i in range(self.num_stage):
            block = getattr(self, f"block{i+1}", None)
            y = block(y)
        y = self.pool(y)
        y = self.fc(y)

        return y
