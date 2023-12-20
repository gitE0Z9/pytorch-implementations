import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_channel: int):
        super(PrototypicalNetwork, self).__init__()
        self.conv_1 = nn.Sequential(ConvBnRelu(input_channel, 32, 10), nn.MaxPool2d(2))
        self.conv_2 = nn.Sequential(ConvBnRelu(32, 64, 7), nn.MaxPool2d(2))
        self.conv_3 = nn.Sequential(ConvBnRelu(64, 128, 3), nn.MaxPool2d(2))
        self.conv_4 = nn.Sequential(ConvBnRelu(128, 256, 3), nn.MaxPool2d(2))

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.conv_4(y)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # share embedding
        y = self.feature_extract(x)

        # flatten space
        y = y.view(x.size(0), x.size(1), -1)

        return y
