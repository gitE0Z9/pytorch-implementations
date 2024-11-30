import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import InvertedResidualBlock


class MobileNetV2(nn.Module):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """MobileNet version 2 [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        super().__init__()

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                int(32 * width_multiplier),
                stride=2,
            ),
            # middle layers
            *self.build_middle_layers(width_multiplier),
            # final layers
            Conv2dNormActivation(
                int(320 * width_multiplier),
                int(1280 * width_multiplier) if width_multiplier > 1 else 1280,
                1,
            ),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                int(1280 * width_multiplier) if width_multiplier > 1 else 1280,
                output_size,
            ),
        )

    def build_middle_layers(
        self,
        width_multiplier: float = 1,
    ) -> list[nn.Module]:
        """Build middle layers.

        Args:
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """

        # input_channel, output_channel, stride, expansion_factor, num_layer
        configs = [
            (32, 16, 1, 1, 1),
            (16, 24, 2, 6, 2),
            (24, 32, 2, 6, 3),
            (32, 64, 2, 6, 4),
            (64, 96, 1, 6, 3),
            (96, 160, 2, 6, 3),
            (160, 320, 1, 6, 1),
        ]
        middle_layers = []
        for (
            in_c,
            out_c,
            stride,
            expansion_ratio,
            num_layer,
        ) in configs:
            middle_layers.extend(
                [
                    InvertedResidualBlock(
                        int((in_c if l == 0 else out_c) * width_multiplier),
                        int(out_c * width_multiplier),
                        stride=stride if l == 0 else 1,
                        expansion_ratio=expansion_ratio,
                    )
                    for l in range(num_layer)
                ]
            )

        return middle_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)
