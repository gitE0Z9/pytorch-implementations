import math
import torch
import torch.nn.functional as F
from torch import nn


class PUPDecoder(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
    ):
        """Progressively upsampling"""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, 256, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, output_size, 3, padding=1),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x = x.pop()
        num_patch = int(math.sqrt(x.size(1)))
        # b, s, d => b, d, h, w
        y = x.transpose(-1, -2).unflatten(-1, (num_patch, num_patch))
        # b, d, h, w => b, o, h, w
        return self.block(y)


class MLADecoder(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
    ):
        """Multi-Level feature aggregation"""
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(input_channel, 256, 3, padding=1),
                nn.Conv2d(input_channel, 256, 3, padding=1),
                nn.Conv2d(input_channel, 256, 3, padding=1),
                nn.Conv2d(input_channel, 256, 3, padding=1),
            ]
        )

        self.necks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.Conv2d(256, 256, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.Conv2d(256, 256, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.Conv2d(256, 256, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.Conv2d(256, 256, 3, padding=1),
                ),
            ]
        )

        self.head = nn.Sequential(
            nn.Conv2d(4 * 256, output_size, 3, padding=1),
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        num_patch = int(math.sqrt(x[0].size(1)))
        features = []

        # b, s, d => b, c, h, w
        y = x.pop().transpose(-1, -2).unflatten(-1, (num_patch, num_patch))
        y = self.blocks[0](y)
        features.append(F.interpolate(self.necks[0](y), scale_factor=4))
        for block, neck in zip(self.blocks[1:], self.necks[1:]):
            z = x.pop().transpose(-1, -2).unflatten(-1, (num_patch, num_patch))
            z = block(z)
            y = y + z
            features.append(F.interpolate(neck(y), scale_factor=4))

        return F.interpolate(self.head(torch.cat(features, 1)), scale_factor=4)
