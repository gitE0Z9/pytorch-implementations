from typing import Literal, Sequence

import torch
from torch import nn

from torchvision.ops import Conv2dNormActivation


class InceptionBlockV2(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channels: Sequence[int | Sequence[int]],
        kernels: Sequence[int] = (1, 3, 5),
        stride: int = 1,
        pooling_type: Literal["mean", "max"] = "mean",
        pooling_kernel: int = 3,
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

        branches = []
        for out_c_in_branch, kernel in zip(output_channels, kernels):
            if kernel == 1 and out_c_in_branch <= 0:
                continue

            layers = [
                Conv2dNormActivation(
                    input_channel,
                    (out_c_in_branch if kernel == 1 else out_c_in_branch[0]),
                    1,
                )
            ]

            if kernel > 1:
                num3x3 = kernel // 2
                for i in range(num3x3):
                    layers.append(
                        Conv2dNormActivation(
                            out_c_in_branch[i],
                            out_c_in_branch[i + 1],
                            3,
                        ),
                    )

            layers[-1][0].stride = stride

            branches.append(nn.Sequential(*layers))

        pooling_branch = nn.Sequential(
            nn.MaxPool2d(
                pooling_kernel,
                stride=self.stride,
                padding=pooling_kernel // 2,
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
