import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import ResBlock

# input, output, number_block
CONFIGS = {
    0.5: [
        [24, 48, 4],
        [48, 96, 8],
        [96, 192, 4],
    ],
    1: [
        [24, 116, 4],
        [116, 232, 8],
        [232, 464, 4],
    ],
    1.5: [
        [24, 176, 4],
        [176, 352, 8],
        [352, 704, 4],
    ],
    2: [
        [24, 244, 4],
        [244, 488, 8],
        [488, 976, 4],
    ],
}


class ShuffleNetV2(nn.Module):
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
            groups (int, optional): number of groups of channel shuffle. Defaults to 1.
            scale_factor (float, optional): scale factor s. Defaults to 1.
            configs (list[list[int]], optional): configs for shufflenet, key is number of groups. Defaults to CONFIGS.
        """
        super(ShuffleNetV2, self).__init__()
        self.config = configs[scale_factor]
        self.num_stage = len(self.config)
        # for identity check
        self.scale_factor = scale_factor

        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.config[0][0],
                3,
                stride=2,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.build_blocks(groups)
        self.conv = Conv2dNormActivation(
            self.config[-1][1],
            1024 if scale_factor < 2 else 2048,
            1,
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(
            1024 if scale_factor < 2 else 2048,
            output_size,
        )

    def build_blocks(self, groups: int = 1):
        """build blocks

        Args:
            groups (int, optional): number of groups of channel shuffle. Defaults to 1.
        """
        for block_index, (
            input_channel,
            output_channel,
            num_layer,
        ) in enumerate(self.config):
            layers = [
                ResBlock(
                    input_channel if layer_index == 0 else output_channel,
                    output_channel,
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
        y = self.conv(y)
        y = self.pool(y)
        y = self.fc(y)

        return y
