import torch
from torch import nn

from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.conv import ConvBnRelu


class TNet(nn.Module):

    def __init__(self, base_channel: int):
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBnRelu(base_channel, 64, 1, dimension="1d"),
            ConvBnRelu(64, 128, 1, dimension="1d"),
            ConvBnRelu(128, 1024, 1, dimension="1d"),
            FlattenFeature("max", "1d"),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, base_channel * base_channel),
            nn.Unflatten(-1, (base_channel, base_channel)),
        )

        self.blocks[-2].weight.data.mul_(0)
        self.blocks[-2].bias.data.copy_(torch.eye(base_channel).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TransformModule(nn.Module):
    def __init__(self, base_channel: int):
        super().__init__()
        self.block = TNet(base_channel)

    def forward(
        self, x: torch.Tensor, output_affine: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # h, h
        t = self.block(x)
        # h, h x h, N => h, N
        y = torch.bmm(t, x)

        if output_affine:
            return y, t
        else:
            return y
