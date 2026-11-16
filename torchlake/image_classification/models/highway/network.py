import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import HighwayLayer


class HighwayBlock(nn.Module):

    def __init__(self, input_channel: int, output_channel: int, kernel: int):
        super().__init__()
        self.layers = nn.Sequential(
            HighwayLayer(
                nn.Sequential(
                    Conv2dNormActivation(output_channel, output_channel, kernel)
                ),
                nn.Sequential(
                    Conv2dNormActivation(output_channel, output_channel, kernel)
                ),
            ),
        )
        if input_channel != output_channel:
            self.layers.insert(
                0, Conv2dNormActivation(input_channel, output_channel, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
