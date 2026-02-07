import torch
from torch import nn

from torchlake.common.models.conv import ConvBNReLU


class DenseBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        growth_rate: int,
        kernel: int = 3,
        num_layer: int = 4,
        dropout_prob: float = 0.2,
    ):
        self.growth_rate = growth_rate
        self.num_layer = num_layer
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(num_layer):
            layer = nn.Sequential(
                ConvBNReLU(
                    input_channel + i * growth_rate,
                    growth_rate,
                    kernel,
                    padding=kernel // 2,
                    conv_last=True,
                ),
                nn.Dropout(dropout_prob),
            )
            nn.init.kaiming_uniform_(layer[0].conv.weight.data, nonlinearity="relu")
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features = []

        # y = self.layers[0](x)
        # features.append(y)
        # for layer in self.layers[1:]:
        #     z = torch.cat((y, x), 1)
        #     y = layer(z)
        #     features.append(y)

        # return torch.cat(features, 1)
        y = x
        for layer in self.layers:
            y = torch.cat((layer(y), y), 1)

        return y


class TransitionDown(nn.Module):

    def __init__(
        self,
        input_channel: int,
        kernel: int = 1,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(
                input_channel,
                input_channel,
                kernel,
                padding=kernel // 2,
                conv_last=True,
            ),
            nn.Dropout(dropout_prob),
            nn.MaxPool2d(2, 2),
        )
        nn.init.kaiming_uniform_(self.layers[0].conv.weight.data, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
