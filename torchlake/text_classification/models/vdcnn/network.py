import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class KmaxPool1d(nn.Module):

    def __init__(self, topk: int):
        super(KmaxPool1d, self).__init__()
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.topk(self.topk, -1)[0]


class Block(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
    ):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(
                input_channel,
                output_channel,
                kernel,
                padding=kernel // 2,
                dimension="1d",
            ),
            ConvBnRelu(
                output_channel,
                output_channel,
                kernel,
                padding=kernel // 2,
                dimension="1d",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
