import torch
from torch import nn


class UpSampling(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
    ):
        super(UpSampling, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, 2, stride=2),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
