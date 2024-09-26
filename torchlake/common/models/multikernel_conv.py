from typing import Literal
import torch
from torch import nn
from .conv import ConvBnRelu
from .flatten import FlattenFeature


class MultiKernelConvModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channels: int | list[int],
        kernels: list[int],
        disable_padding: bool = False,
        activation: nn.Module | None = nn.ReLU(True),
        dimension: Literal["1d", "2d", "3d"] = "2d",
        reduction: Literal["mean", "max", "none"] = "none",
        concat_output: bool = False,
    ):
        super(MultiKernelConvModule, self).__init__()
        if isinstance(output_channels, int):
            output_channels = [output_channels] * len(kernels)

        if reduction in ["mean", "max"]:
            self.flatten = FlattenFeature(
                reduction=reduction,
                dimension=dimension,
                start_dim=2,
            )

        self.concat_output = concat_output

        self.cnns = nn.ModuleList(
            [
                ConvBnRelu(
                    input_channel,
                    output_channel,
                    kernel,
                    padding=(
                        (
                            (k // 2 for k in kernel)
                            if isinstance(kernel, tuple)
                            else kernel // 2
                        )
                        if not disable_padding
                        else 0
                    ),
                    enable_bn=False,
                    activation=activation,
                    dimension=dimension,
                )
                for output_channel, kernel in zip(output_channels, kernels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        y = [layer(x) for layer in self.cnns]

        flatten = getattr(self, "flatten", None)
        if flatten is not None:
            y = [flatten(output).squeeze(-1) for output in y]

        if self.concat_output:
            y = torch.cat(y, 1)

        return y
