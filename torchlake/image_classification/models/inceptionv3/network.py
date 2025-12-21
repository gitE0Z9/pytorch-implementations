from itertools import pairwise
from typing import Literal, Sequence

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.flatten import FlattenFeature


class AsymmetricConv2d(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channels: int | Sequence[int],
        kernel: int,
        stride: int = 1,
        mode: Literal["sequential", "parallel"] = "sequential",
        kernel_first: bool = True,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channels = output_channels
        self.kernel = kernel
        self.stride = stride
        self.mode = mode
        self.kernel_first = kernel_first

        if mode == "sequential":
            self.layers = nn.Sequential(
                Conv2dNormActivation(
                    input_channel,
                    output_channels[0],
                    kernel_size=(kernel, 1) if kernel_first else (1, kernel),
                ),
                Conv2dNormActivation(
                    output_channels[0],
                    output_channels[1],
                    kernel_size=(1, kernel) if kernel_first else (kernel, 1),
                ),
            )
        else:
            self.layers = nn.ModuleList(
                [
                    Conv2dNormActivation(
                        input_channel,
                        output_channels[0],
                        kernel_size=(kernel, 1) if kernel_first else (1, kernel),
                    ),
                    Conv2dNormActivation(
                        input_channel,
                        output_channels[1],
                        kernel_size=(1, kernel) if kernel_first else (kernel, 1),
                    ),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sequential":
            y = x
            for layer in self.layers:
                y = layer(y)
        else:
            y = torch.cat(tuple(layer(x) for layer in self.layers), 1)

        return y


class InceptionBlockV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channels: Sequence[int | Sequence[int]],
        kernels: Sequence[int] = (1, 3, 5),
        stride: int = 1,
        pooling_type: Literal["mean", "max"] = "mean",
        pooling_kernel: int = 3,
        mode: Literal["sequential", "parallel"] = "sequential",
    ):
        super().__init__()
        assert (
            len(output_channels) == len(kernels) + 1
        ), "the length of output channels has to be one more by the length of kernels"

        self.input_channel = input_channel
        self.output_channels = output_channels
        self.kernels = kernels
        self.stride = stride
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.mode = mode

        branches = []
        for out_c_in_branch, kernel_in_branch in zip(output_channels, kernels):
            if isinstance(out_c_in_branch, int) and isinstance(kernel_in_branch, int):
                layers = [
                    Conv2dNormActivation(
                        input_channel,
                        out_c_in_branch,
                        kernel_in_branch,
                    )
                ]
            else:
                layers = []
                for (in_c, out_c), k in zip(
                    pairwise((input_channel, *out_c_in_branch)),
                    kernel_in_branch,
                ):
                    if isinstance(in_c, Sequence):
                        if layers[-1].mode == "sequential":
                            in_c = in_c[-1]
                        else:
                            in_c = sum(in_c)

                    if isinstance(k, Sequence):
                        k, kernel_first = k
                        layers.append(
                            AsymmetricConv2d(
                                in_c,
                                out_c,
                                k,
                                mode=mode,
                                kernel_first=kernel_first,
                            )
                        )
                    else:
                        layers.append(Conv2dNormActivation(in_c, out_c, k))

            if stride > 1:
                layers[-1][0].stride = stride
                layers[-1][0].padding = 0

            branches.append(nn.Sequential(*layers))

        pooling_branch = nn.Sequential(
            nn.MaxPool2d(
                pooling_kernel,
                stride=self.stride,
                padding=pooling_kernel // 2 if self.stride == 1 else 0,
            ),
        )
        if output_channels[-1] > 0:
            pooling_branch.append(
                Conv2dNormActivation(
                    input_channel,
                    output_channels[-1],
                    1,
                ),
            )
        branches.append(pooling_branch)

        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [branch(x) for branch in self.branches]

        y = torch.cat(y, 1)

        return y


class AuxiliaryClassifierV3(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dims: Sequence[int],
        kernel: int,
        pooling_type: Literal["mean", "max"] = "mean",
        pooling_kernel: int = 5,
        dropout_prob: float = 0,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_size = output_size
        self.hidden_dims = hidden_dims
        self.kernel = kernel
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.dropout_prob = dropout_prob

        self.layers = nn.Sequential(
            nn.AvgPool2d(pooling_kernel, stride=3),
            Conv2dNormActivation(input_channel, hidden_dims[0], 1),
            Conv2dNormActivation(hidden_dims[0], hidden_dims[1], kernel, padding=0),
            FlattenFeature(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(hidden_dims[1], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
