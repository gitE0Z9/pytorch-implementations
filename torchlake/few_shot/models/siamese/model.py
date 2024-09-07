import torch
from torch import nn
from torchlake.common.models import FlattenFeature
from torchvision.ops import Conv2dNormActivation


class SiameseNetwork(nn.Module):
    def __init__(self, input_channel: int):
        super(SiameseNetwork, self).__init__()
        self.block1 = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                32,
                10,
                padding=0,
            ),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            Conv2dNormActivation(
                32,
                64,
                7,
                padding=0,
            ),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            Conv2dNormActivation(
                64,
                128,
                3,
                padding=0,
            ),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            FlattenFeature(reduction=None),
            nn.Linear(9 * 9 * 128, 256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256, 1)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for i in range(1, 4):
            y = getattr(self, f"block{i}")(y)

        return self.linear(y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # share embedding
        xprime = self.feature_extract(x)
        yprime = self.feature_extract(y)

        # compare both flatten dimension
        y = torch.abs(xprime - yprime)

        # compute similarity
        return self.fc(y)
