import torch
from torch import nn

from .network import HighwayLayer


class HighwayNetwork(nn.Module):

    def __init__(self, configs: list[list[int]], output_size: int = 1):
        super(HighwayNetwork, self).__init__()
        self.blocks = self.build_blocks(configs)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(configs[-1][2], output_size)

    def build_blocks(self, configs: list[list[int]]) -> list[nn.Module]:
        blocks = []

        for i, (layer_type, in_channel, out_channel, kernel_size) in enumerate(configs):
            if layer_type == "c":
                layer = HighwayLayer(in_channel, out_channel, kernel_size)
                blocks.append(layer)
            elif layer_type == "p":
                pooling = nn.MaxPool2d(2, 2)
                blocks.append(pooling)
            else:
                pass

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        y = self.pool(y)
        return self.fc(y)
