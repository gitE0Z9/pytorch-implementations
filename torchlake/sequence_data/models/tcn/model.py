import torch
from torch import nn

from .network import BottleneckBlock


class Tcn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super(Tcn, self).__init__()

        self.layers = nn.Sequential(
            *[
                BottleneckBlock(
                    input_dim if i == 0 else num_channels[i - 1],
                    num_channels[i],
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * 2**i,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(len(num_channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
