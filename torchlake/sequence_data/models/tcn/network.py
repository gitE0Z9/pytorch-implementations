import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from torchlake.common.models.conv import CausalConv1d


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
            nn.Sequential(
                weight_norm(
                    CausalConv1d(
                        input_dim,
                        output_dim,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                ),
                nn.ReLU(True),
            ),
            nn.Sequential(
                weight_norm(
                    CausalConv1d(
                        output_dim,
                        output_dim,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                ),
                nn.ReLU(True),
            ),
        )

        if dropout > 0:
            for i in range(len(self.layers)):
                self.layers[i].append(nn.Dropout1d(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
