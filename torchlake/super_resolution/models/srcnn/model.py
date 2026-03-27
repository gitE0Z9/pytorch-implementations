from itertools import pairwise
from typing import Self, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.super_resolution.models.srcnn.network import init_conv_srcnn_style


class SRCNN(ModelBase):
    def __init__(
        self,
        input_channel: int,
        kernels: Sequence[int],
        hidden_dims: Sequence[int],
    ):
        self.kernels = kernels
        self.hidden_dims = hidden_dims
        super().__init__(input_channel, input_channel)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential()
        for channels, kernel in zip(
            pairwise((input_channel, *self.hidden_dims)),
            self.kernels[:-1],
        ):
            self.foot.append(
                nn.Sequential(
                    nn.Conv2d(*channels, kernel),
                    nn.ReLU(True),
                )
            )
            init_conv_srcnn_style(self.foot[-1][0])

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dims[-1], output_size, self.kernels[-1]),
        )
        init_conv_srcnn_style(self.head[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.foot(x)
        else:
            for mod in self.foot:
                x = F.pad(
                    x,
                    (
                        mod[0].kernel_size[1] // 2,
                        mod[0].kernel_size[1] // 2,
                        mod[0].kernel_size[0] // 2,
                        mod[0].kernel_size[0] // 2,
                    ),
                    mode="replicate",
                )
                x = mod(x)

        if not self.training:
            x = F.pad(
                x,
                (
                    self.kernels[-1] // 2,
                    self.kernels[-1] // 2,
                    self.kernels[-1] // 2,
                    self.kernels[-1] // 2,
                ),
                mode="replicate",
            )
        return self.head(x)
