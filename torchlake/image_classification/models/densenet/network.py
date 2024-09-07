import torch
from torch import nn
from torchlake.common.models import ConvBnRelu


class DenseBlock(nn.Module):

    def __init__(
        self,
        input_channel: int = 3,
        num_layer: int = 1,
        growth_rate: int = 32,
        expansion_ratio: int = 4,
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(
                    ConvBnRelu(
                        growth_rate * i + input_channel,
                        expansion_ratio * growth_rate,
                        1,
                        conv_last=True,
                    ),
                    ConvBnRelu(
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

    def __init__(
        self,
        input_channel: int = 3,
        output_channel: int = 1,
    ):
        super(TransitionBlock, self).__init__()
        self.conv = ConvBnRelu(
            input_channel,
            output_channel,
            1,
            conv_last=True,
        )

        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return self.pool(y)
