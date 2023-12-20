import torch
from torch import nn
from .network import ResBlock


class ResNet50(nn.Module):
    def __init__(self, input_channel: int = 3, label_size: int = 1):
        super(ResNet50, self).__init__()
        self.foot = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2),
        )
        self.block1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(256, 64),
            ResBlock(256, 64),
        )
        self.block2 = nn.Sequential(
            ResBlock(256, 128),
            ResBlock(512, 128),
            ResBlock(512, 128),
            ResBlock(512, 128),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            ResBlock(512, 256),
            ResBlock(1024, 256),
            ResBlock(1024, 256),
            ResBlock(1024, 256),
            ResBlock(1024, 256),
            ResBlock(1024, 256),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            ResBlock(1024, 512),
            ResBlock(2048, 512),
            ResBlock(2048, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(nn.Linear(2048, label_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        return y
