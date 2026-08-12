import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


def pad_on_left(x: torch.Tensor, offset: int):
    return F.pad(x, (offset, 0))


class CausalConv1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = padding
        self.layer = weight_norm(
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size,
                stride=stride,
                dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(pad_on_left(x, self.padding))


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        dropout: float = 0,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(
                input_dim,
                output_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(True),
            nn.Dropout(dropout),
            CausalConv1d(
                output_dim,
                output_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

        if dropout == 0:
            self.layers.pop(5)
            self.layers.pop(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
