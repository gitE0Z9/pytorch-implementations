import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class ConvInRelu(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super(ConvInRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.InstanceNorm2d(output_channel),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownSampling(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super(DownSampling, self).__init__()
        self.model = nn.Sequential(
            ConvBnRelu(input_channel, output_channel, 3, padding=1),
            ConvBnRelu(output_channel, output_channel, 3, padding=1),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UpSampling(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super(UpSampling, self).__init__()
        self.model = nn.Sequential(
            ConvBnRelu(input_channel, output_channel, 3, padding=1),
            ConvBnRelu(output_channel, output_channel, 3, padding=1),
            nn.ConvTranspose2d(output_channel, output_channel // 2, 2, stride=2),
        )

    def forward(self, x: torch.Tensor, feautre: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.model(x), feautre], dim=1)
