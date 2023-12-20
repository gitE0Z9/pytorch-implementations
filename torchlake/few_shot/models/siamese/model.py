import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class SiameseNetwork(nn.Module):
    def __init__(self, input_channel: int):
        super(SiameseNetwork, self).__init__()
        self.conv_1 = nn.Sequential(ConvBnRelu(input_channel, 32, 10), nn.MaxPool2d(2))
        self.conv_2 = nn.Sequential(ConvBnRelu(32, 64, 7), nn.MaxPool2d(2))
        self.conv_3 = nn.Sequential(ConvBnRelu(64, 128, 3), nn.MaxPool2d(2))
        self.linear = nn.Sequential(nn.Linear(9 * 9 * 128, 256), nn.ReLU(inplace=True))
        self.clf = nn.Linear(256, 1)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = y.view(-1, 9 * 9 * 128)
        y = self.linear(y)

        return y

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # share embedding
        xprime = self.feature_extract(x)
        yprime = self.feature_extract(y)

        # compare both flatten dimension
        dif = torch.abs(xprime - yprime)

        # compute similarity
        dif = self.clf(dif)

        return dif
