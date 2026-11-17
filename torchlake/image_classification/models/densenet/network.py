import torch
from torch import nn
from torchlake.common.models import ConvBNReLU


class DenseBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        num_layer: int,
        growth_rate: int = 32,
        expansion_ratio: int = 4,
    ):
        super().__init__()
        self.output_channel = input_channel + growth_rate * num_layer
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(
                    ConvBNReLU(
                        growth_rate * i + input_channel,
                        expansion_ratio * growth_rate,
                        1,
                        conv_last=True,
                    ),
                    ConvBNReLU(
                        expansion_ratio * growth_rate,
                        growth_rate,
                        3,
                        padding=1,
                        conv_last=True,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], 1)

        return x


class TransitionBlock(nn.Module):

    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(
                input_channel,
                output_channel,
                1,
                conv_last=True,
            ),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
