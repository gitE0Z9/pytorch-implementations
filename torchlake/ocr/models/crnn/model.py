import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import RecurrnetBlock


class Crnn(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, nclass: int):
        super(Crnn, self).__init__()

        self.cnn = nn.Sequential(
            Conv2dNormActivation(input_dim, 64, 3, norm_layer=None),  # B, 64, 32, W
            nn.MaxPool2d(2, 2),  # B, 64, 16, W//2
            Conv2dNormActivation(64, 128, 3, norm_layer=None),  # B, 128, 16, W//2
            nn.MaxPool2d(2, 2),  # B, 128, 8, W//4
            Conv2dNormActivation(128, 256, 3, norm_layer=None),  # B, 256, 8, W//4
            Conv2dNormActivation(256, 256, 3, norm_layer=None),
            nn.MaxPool2d((2, 1), (2, 1)),  # B, 256, 4, W//4
            Conv2dNormActivation(256, 512, 3),  # B, 512, 4, W//4
            Conv2dNormActivation(512, 512, 3),
            nn.MaxPool2d((2, 1), (2, 1)),  # B, 512, 2, W//4
            Conv2dNormActivation(512, 512, 2, padding=0),  # B, 512, 1, W//4-1
        )

        self.rnn = nn.Sequential(
            RecurrnetBlock(512, hid_dim, nclass, 2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        _, _, h, _ = x.shape
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2).permute(2, 0, 1)  # w, b, c
        o = self.rnn(x)

        return o
