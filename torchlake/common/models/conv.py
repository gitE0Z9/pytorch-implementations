from typing import Literal

import torch
from torch import nn


class ConvBNReLU(nn.Module):
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
        dimension: Literal["1d", "2d", "3d"] = "2d",
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        deconvolution: bool = False,
        output_padding: int = 0,
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
        super().__init__()
        self.conv_last = conv_last

        if deconvolution:
            # O = (I-1) * S - 2P + D(K-1) + OP + 1
            conv_cls = {
                "1d": nn.ConvTranspose1d,
                "2d": nn.ConvTranspose2d,
                "3d": nn.ConvTranspose3d,
            }[dimension]
        else:
            # O = ceil(I - K + 1 + 2P / S)
            conv_cls = {
                "1d": nn.Conv1d,
                "2d": nn.Conv2d,
                "3d": nn.Conv3d,
            }[dimension]

        bn_cls = {
            "1d": nn.BatchNorm1d,
            "2d": nn.BatchNorm2d,
            "3d": nn.BatchNorm3d,
        }[dimension]

        kwargs = {
            "in_channels": input_channel,
            "out_channels": output_channel,
            "kernel_size": kernel,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": group,
            "bias": not enable_bn,
        }
        if deconvolution:
            kwargs["output_padding"] = output_padding
        else:
            kwargs["padding_mode"] = padding_mode

        self.conv = conv_cls(**kwargs)
        self.bn = (
            bn_cls(output_channel if not conv_last else input_channel)
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


class ConvInReLU(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        group: int = 1,
        enable_in: bool = True,
        activation: nn.Module | None = nn.LeakyReLU(0.2),
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
            enable_in (bool, optional): enable instance normalization. Defaults to True.
            activation (nn.Module | None, optional): activation function. Defaults to nn.LeakyReLU(0.2).
            conv_last (bool, optional): change order to BN-ReLU-Conv. Defaults to False.
            dimension (Literal["1d"] | Literal["2d"] | Literal["3d"], optional): 1d, 2d or 3d. Defaults to "2d".
        """
        super().__init__()

        pad_cls = {
            "1d": nn.ReflectionPad1d,
            "2d": nn.ReflectionPad2d,
            "3d": nn.ReflectionPad3d,
        }[dimension]

        conv_cls = {
            "1d": nn.Conv1d,
            "2d": nn.Conv2d,
            "3d": nn.Conv3d,
        }[dimension]

        norm_cls = {
            "1d": nn.InstanceNorm1d,
            "2d": nn.InstanceNorm2d,
            "3d": nn.InstanceNorm3d,
        }[dimension]

        if padding is None:
            padding = kernel // 2
        self.pad = pad_cls(padding)
        self.conv = conv_cls(
            input_channel,
            output_channel,
            kernel,
            stride,
            padding=0,
            dilation=dilation,
            groups=group,
            bias=not enable_in,
        )
        self.norm = norm_cls(output_channel, affine=True) if enable_in else enable_in
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pad(x)
        y = self.conv(y)

        if self.norm:
            y = self.norm(y)

        if self.activation is not None:
            y = self.activation(y)

        return y
