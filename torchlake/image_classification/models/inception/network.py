from typing import Literal, Sequence

import torch
from torch import nn

from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import FlattenFeature


class InceptionBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channels: Sequence[int | Sequence[int]],
        kernels: Sequence[int] = (1, 3, 5),
        pooling_type: Literal["mean", "max"] = "max",
        pooling_kernel: int = 3,
    ):
        super().__init__()
        assert (
            len(output_channels) == len(kernels) + 1
        ), "the length of output channels has to be one more by the length of kernels"

        self.input_channel = input_channel
        self.output_channels = output_channels
        self.kernels = kernels
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel

        branches = []
        for out_c_in_branch, kernel in zip(output_channels, kernels):
            layers = [
                Conv2dNormActivation(
                    input_channel,
                    (
                        out_c_in_branch
                        if isinstance(out_c_in_branch, int)
                        else out_c_in_branch[0]
                    ),
                    1,
                    norm_layer=None,
                )
            ]
            if kernel > 1:
                layers.append(
                    Conv2dNormActivation(
                        out_c_in_branch[0],
                        out_c_in_branch[1],
                        kernel,
                        norm_layer=None,
                    ),
                )
            branches.append(nn.Sequential(*layers))

        branches.append(
            nn.Sequential(
                nn.MaxPool2d(pooling_kernel, stride=1, padding=pooling_kernel // 2),
                Conv2dNormActivation(
                    input_channel,
                    output_channels[-1],
                    kernel_size=1,
                    norm_layer=None,
                ),
            )
        )

        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [branch(x) for branch in self.branches]

        y = torch.cat(y, 1)

        return y


class AuxiliaryClassifier(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dims: Sequence[int],
        pooling_type: Literal["mean", "max"] = "mean",
        pooling_kernel: int = 5,
        dropout_prob: float = 0.7,
        enable_bn: bool = True,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_size = output_size
        self.hidden_dims = hidden_dims
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.dropout_prob = dropout_prob
        self.enable_bn = enable_bn

        self.layers = nn.Sequential(
            nn.AvgPool2d(pooling_kernel, stride=3),
            Conv2dNormActivation(
                input_channel,
                hidden_dims[0],
                1,
                norm_layer=torch.nn.BatchNorm2d if enable_bn else None,
            ),
            FlattenFeature(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[1], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
