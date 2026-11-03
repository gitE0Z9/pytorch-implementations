import math
from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from torchlake.common.models.residual import ResBlock
from torchlake.sequence_data.models.tcn import CausalConv1d


def glu_kaming_init(w: torch.Tensor, b: torch.Tensor | None = None):
    nn.init.normal_(
        w,
        0,
        math.sqrt(4 / math.prod(w.shape[1:])),
    )
    if b is not None:
        nn.init.zeros_(b)


class GatedCausalConv1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        preactivation: bool = False,
    ):
        super().__init__()
        layer = CausalConv1d(
            input_dim if not preactivation else input_dim // 2,
            output_dim * 2 if not preactivation else output_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            enable_weight_norm=False,
        )
        glu_kaming_init(layer.layer.weight, layer.layer.bias)
        weight_norm(layer.layer)

        self.layers = nn.Sequential(layer)
        self.layers.insert(0 if preactivation else 1, nn.GLU(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Bottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        kernels: Sequence[int],
        stride: int = 1,
        preactivation: bool = False,
    ):
        super().__init__()
        self.layers = nn.Sequential()

        in_c = input_dim
        for out_c, kernel in zip(hidden_dims, kernels):
            self.layers.append(
                nn.Sequential(
                    GatedCausalConv1d(
                        in_c,
                        out_c,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel - 1,
                        preactivation=preactivation,
                    ),
                )
            )
            in_c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResBlock(ResBlock):
    def build_shortcut(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
    ) -> nn.Module:
        """build shortcut

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
        """
        if input_channel == output_channel and stride == 1:
            return nn.Identity()

        layer = CausalConv1d(
            input_channel,
            output_channel,
            1,
            stride=stride,
            enable_weight_norm=False,
        )
        glu_kaming_init(layer.layer.weight, layer.layer.bias)
        weight_norm(layer.layer)

        return nn.Sequential(layer)
