import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class ResBlock(nn.Module):
    def __init__(self, input_channel: int, block_base_channel: int):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBnRelu(input_channel, block_base_channel, 1),
            ConvBnRelu(block_base_channel, block_base_channel, 3, padding=1),
            ConvBnRelu(block_base_channel, block_base_channel * 4, 1),
        )

        self.downsample = (
            nn.Identity()
            if input_channel == block_base_channel * 4
            else nn.Sequential(ConvBnRelu(input_channel, block_base_channel * 4, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.downsample(x)
