import torch
from torch import nn
import math
from torchlake.common.network import ConvBnRelu


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(DoubleConv, self).__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch, 3, padding=1)
        self.conv2 = ConvBnRelu(out_ch, out_ch, 3, padding=1)

        # nn.init.normal_(self.conv1.conv.weight, 0, math.sqrt(2 / 9 / in_ch))
        # nn.init.normal_(self.conv2.conv.weight, 0, math.sqrt(2 / 9 / out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class DownSampling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(DownSampling, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.convs = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.convs(y)
        return y


class UpSampling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.convs = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y_pad, x_pad = feature.size(3) - y.size(3), feature.size(2) - y.size(2)
        y = nn.ReflectionPad2d((0, 0, y_pad // 2, x_pad // 2))(y)
        y = torch.cat([feature, y], dim=1)
        y = self.convs(y)
        return y
