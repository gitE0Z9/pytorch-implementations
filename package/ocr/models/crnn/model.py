import torch
from common.block import ConvBnRelu
from ocr.models.crnn.network import RecurrnetBlock
from torch import nn


class Crnn(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, nclass: int):
        super(Crnn, self).__init__()

        self.cnn = nn.Sequential(
            ConvBnRelu(input_dim, 64, 3, 1, 1, bn=False),  # B, 64, 32, W
            nn.MaxPool2d(2, 2),  # B, 64, 16, W//2
            ConvBnRelu(64, 128, 3, 1, 1, bn=False),  # B, 128, 16, W//2
            nn.MaxPool2d(2, 2),  # B, 128, 8, W//4
            ConvBnRelu(128, 256, 3, 1, 1, bn=False),  # B, 256, 8, W//4
            ConvBnRelu(256, 256, 3, 1, 1, bn=False),
            nn.MaxPool2d((2, 1), (2, 1)),  # B, 256, 4, W//4
            ConvBnRelu(256, 512, 3, 1, 1),  # B, 512, 4, W//4
            ConvBnRelu(512, 512, 3, 1, 1),
            nn.MaxPool2d((2, 1), (2, 1)),  # B, 512, 2, W//4
            ConvBnRelu(512, 512, 2, 1, 0),  # B, 512, 1, W//4-1
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
