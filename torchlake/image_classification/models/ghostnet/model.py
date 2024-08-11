import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import GhostLayer


class GhostNet(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """GhostNet version 1 [1911.11907v2]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        super(GhostNet, self).__init__()

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                int(16 * width_multiplier),
                stride=2,
            ),
            # middle layers
            *self.build_middle_layers(width_multiplier),
            # final layers
            Conv2dNormActivation(
                int(160 * width_multiplier),
                int(960 * width_multiplier),
                1,
            ),
            Conv2dNormActivation(
                int(960 * width_multiplier),
                int(1280 * width_multiplier),
                1,
            ),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            Conv2dNormActivation(
                int(1280 * width_multiplier),
                output_size,
                1,
                norm_layer=None,
                activation_layer=None,
            ),
            nn.Flatten(),
        )

    def build_middle_layers(
        self,
        width_multiplier: float = 1,
    ) -> list[nn.Module]:
        """Build middle layers.

        Args:
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        # input_channel, output_channel, kernel, stride, expansion_size, enable_se
        config = [
            (16, 16, 3, 1, 16, False),
            (16, 24, 3, 2, 48, False),
            (24, 24, 3, 1, 72, False),
            (24, 40, 5, 2, 72, True),
            (40, 40, 5, 1, 120, True),
            (40, 80, 5, 2, 240, False),
            (80, 80, 5, 1, 200, False),
            (80, 80, 5, 1, 184, False),
            (80, 80, 5, 1, 184, False),
            (80, 112, 5, 1, 480, True),
            (112, 112, 5, 1, 672, True),
            (112, 160, 5, 2, 672, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
            (160, 160, 5, 1, 960, False),
            (160, 160, 5, 1, 960, True),
        ]

        middle_layers = []
        for (
            in_c,
            out_c,
            kernel,
            stride,
            expansion_size,
            enable_se,
        ) in config:
            middle_layers.append(
                GhostLayer(
                    int(in_c * width_multiplier),
                    int(out_c * width_multiplier),
                    kernel,
                    stride=stride,
                    s=2,
                    d=3,
                    expansion_size=expansion_size,
                    enable_se=enable_se,
                )
            )

        return middle_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)
