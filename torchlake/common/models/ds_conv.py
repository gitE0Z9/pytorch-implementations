import torch
from torch import nn

from ..network import ConvBnRelu
from .se import SqueezeExcitation2d


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        enable_bn: tuple[bool] = (True, True),
        activation: tuple[nn.Module | None] = (nn.ReLU(True), nn.ReLU(True)),
        enable_se: bool = False,
        reduction_ratio: float = 1,
    ):
        """DepthwiseSeparableConv2d, consist of depthwise separable convolution layer and pointwise convolution layer
        3 -> 1
        input_channel -> input_channel or output_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            kernel (int): kernel size of depthwise separable convolution layer
            stride (int, optional): stride of depthwise separable convolution layer. Defaults to 1.
            padding (tuple[int], optional): padding of both layers. Defaults to (0, 0).
            dilation (tuple[int], optional): dilation of both layers. Defaults to (1, 1).
            enable_bn (tuple[bool], optional): enable_bn of both layers. Defaults to (True, True).
            activation (tuple[nn.Module  |  None], optional): activation of both layers. Defaults to (nn.ReLU(True), nn.ReLU(True)).
            enable_se (bool, optional): enable squeeze and excitation. Defaults to False.
        """
        super(DepthwiseSeparableConv2d, self).__init__()
        self.enable_se = enable_se
        latent_dim = (
            output_channel if input_channel == output_channel else input_channel
        )
        self.depthwise_separable_layer = ConvBnRelu(
            input_channel,
            latent_dim,
            kernel,
            stride,
            padding,
            dilation,
            group=input_channel,
            enable_bn=enable_bn[0],
            activation=activation[0],
        )
        self.pointwise_layer = ConvBnRelu(
            latent_dim,
            output_channel,
            1,
            enable_bn=enable_bn[1],
            activation=activation[1],
        )

        if self.enable_se:
            self.se = SqueezeExcitation2d(output_channel, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise_separable_layer(x)
        y = self.pointwise_layer(x)

        if self.enable_se:
            return self.se(y)

        return y
