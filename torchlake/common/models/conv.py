from typing import Literal

import torch
from torch import nn


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        group: int = 1,
        enable_bn: bool = True,
        activation: nn.Module | None = nn.ReLU(True),
        conv_last: bool = False,
        dimension: Literal["1d"] | Literal["2d"] | Literal["3d"] = "2d",
    ):
        """Custom Conv-BN-ReLU block

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            kernel (int): kernel size of filter
            stride (int, optional): stride. Defaults to 1.
            padding (int, optional): padding. Defaults to 0.
            dilation (int, optional): dilation. Defaults to 1.
            group (int, optional): group. Defaults to 1.
            enable_bn (bool, optional): enable batch normalization. Defaults to True.
            activation (nn.Module | None, optional): activation function. Defaults to nn.ReLU(True).
            conv_last (bool, optional): change order to BN-ReLU-Conv. Defaults to False.
            dimension (Literal["1d"] | Literal["2d"] | Literal["3d"], optional): 1d, 2d or 3d. Defaults to "2d".
        """
        super(ConvBnRelu, self).__init__()
        self.conv_last = conv_last

        conv_class = {
            "1d": nn.Conv1d,
            "2d": nn.Conv2d,
            "3d": nn.Conv3d,
        }[dimension]

        bn_class = {
            "1d": nn.BatchNorm1d,
            "2d": nn.BatchNorm2d,
            "3d": nn.BatchNorm3d,
        }[dimension]

        self.conv = conv_class(
            input_channel,
            output_channel,
            kernel,
            stride,
            padding,
            dilation,
            group,
            bias=not enable_bn,
        )
        self.bn = (
            bn_class(output_channel if not conv_last else input_channel)
            if enable_bn
            else enable_bn
        )
        self.activation = activation or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.conv_last:
            x = self.conv(x)

        if self.bn:
            x = self.bn(x)
        x = self.activation(x)

        if self.conv_last:
            x = self.conv(x)

        return x
