import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super(TemporalConvBlock, self).__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.chomp(y)
        y = self.relu(y)
        y = self.dropout(y)

        return y


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super(BottleneckBlock, self).__init__()

        self.conv1 = TemporalConvBlock(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dropout=dropout,
        )

        self.conv2 = TemporalConvBlock(
            output_dim,
            output_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dropout=dropout,
        )

        self.downsample = (
            nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(y + res)
