import torch
from torch import nn

from ..network import ConvBnRelu


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
        activations: tuple[nn.Module | None] = (nn.ReLU(True), nn.ReLU(True)),
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
            activations (tuple[nn.Module  |  None], optional): activation of both layers. Defaults to (nn.ReLU(True), nn.ReLU(True)).
        """
        super(DepthwiseSeparableConv2d, self).__init__()
        latent_dim = self.get_latent_dim(input_channel, output_channel)

        self.depthwise_separable_layer = ConvBnRelu(
            input_channel,
            latent_dim,
            kernel,
            stride,
            padding,
            dilation,
            group=input_channel,
            enable_bn=enable_bn[0],
            activation=activations[0],
        )
        self.pointwise_layer = ConvBnRelu(
            latent_dim,
            output_channel,
            1,
            enable_bn=enable_bn[1],
            activation=activations[1],
        )

    def get_latent_dim(self, input_channel: int, output_channel: int) -> int:
        latent_dim = (
            output_channel if input_channel == output_channel else input_channel
        )

        return latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise_separable_layer(x)
        y = self.pointwise_layer(y)
        return y
