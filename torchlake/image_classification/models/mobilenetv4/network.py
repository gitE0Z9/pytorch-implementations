from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.utils.numerical import make_divisible


class LayerScale2d(nn.Module):

    def __init__(self, hidden_dim: int, gamma: float = 1e-5):
        """Per-channel learnable scale [2103.17239]

        Args:
            hidden_dim (int): hidden dimension.
            gamma (float, optional): initial value of scale. Defaults to 1e-5.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.full((hidden_dim, 1, 1), gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class FusedInvertedBottleneck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int = 3,
        stride: int = 1,
        expansion_ratio: float = 4,
        activation: type[nn.Module] = nn.ReLU,
        inplace: bool | None = True,
    ):
        """Fused inverted bottleneck [2404.10518]

        Args:
            input_channel (int): input channel size.
            output_channel (int): output channel size.
            kernel (int, optional): kernel size of expansion conv. Defaults to 3.
            stride (int, optional): stride of expansion conv. Defaults to 1.
            expansion_ratio (float, optional): expansion ratio. Defaults to 4.
            activation (type[nn.Module], optional): activation class. Defaults to nn.ReLU.
            inplace (bool, optional): activation should inplace output. Defaults to True.
        """
        super().__init__()
        hidden_dim = make_divisible(input_channel * expansion_ratio, 8)

        # the expansion conv replaces 1x1 conv + depthwise conv
        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                hidden_dim,
                kernel,
                stride,
                activation_layer=activation,
                inplace=inplace,
            ),
            Conv2dNormActivation(
                hidden_dim,
                output_channel,
                1,
                activation_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UniversalInvertedBottleneck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        ds_kernels: Sequence[int] = (0, 3),
        stride: int = 1,
        expansion_ratio: float = 4,
        activation: type[nn.Module] = nn.ReLU,
        inplace: bool | None = True,
        enable_layer_scale: bool = False,
    ):
        """Universal inverted bottleneck (UIB) [2404.10518]
        start dw -> 1x1 expansion -> middle dw -> 1x1 projection

        variants by optional depthwise convs:
            start dw   middle dw
            no         yes         inverted bottleneck (IB)
            yes        no          ConvNext-like
            yes        yes         ExtraDW
            no         no          FFN

        Args:
            input_channel (int): input channel size.
            output_channel (int): output channel size.
            ds_kernels (Sequence[int], optional): kernel size of depthwise convs, 0 means disabled. Defaults to (0, 3).
            stride (int, optional): stride of middle depthwise conv, or of starting depthwise conv if middle is disabled. Defaults to 1.
            expansion_ratio (float, optional): expansion ratio. Defaults to 4.
            activation (type[nn.Module], optional): activation class. Defaults to nn.ReLU.
            inplace (bool, optional): activation should inplace output. Defaults to True.
            enable_layer_scale (bool, optional): enable layer scale. Defaults to False.
        """
        super().__init__()
        assert (
            len(ds_kernels) == 2
        ), "kernels of both depthwise convs have to be defined"
        if stride > 1:
            assert any(ds_kernels), "downsampling requires a depthwise conv"

        hidden_dim = make_divisible(input_channel * expansion_ratio, 8)

        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                hidden_dim,
                1,
                activation_layer=activation,
                inplace=inplace,
            ),
            Conv2dNormActivation(
                hidden_dim,
                output_channel,
                1,
                activation_layer=None,
            ),
        )

        if ds_kernels[0]:
            self.layers.insert(
                0,
                Conv2dNormActivation(
                    input_channel,
                    input_channel,
                    ds_kernels[0],
                    1 if ds_kernels[1] else stride,
                    groups=input_channel,
                    activation_layer=None,
                ),
            )

        if ds_kernels[1]:
            self.layers.insert(
                -1,
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    ds_kernels[1],
                    stride,
                    groups=hidden_dim,
                    activation_layer=activation,
                    inplace=inplace,
                ),
            )

        if enable_layer_scale:
            self.layers.append(LayerScale2d(output_channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MobileMQA(nn.Module):

    def __init__(
        self,
        input_channel: int,
        num_head: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        kv_stride: int = 1,
        dw_kernel: int = 3,
        enable_layer_scale: bool = False,
    ):
        """Mobile multi-query attention block [2404.10518]
        queries have num_head heads, while key and value are shared across heads
        and spatially downsampled by a strided depthwise conv when kv_stride > 1.

        Args:
            input_channel (int): input channel size.
            num_head (int, optional): number of query heads. Defaults to 8.
            key_dim (int, optional): dimension of query and key per head. Defaults to 64.
            value_dim (int, optional): dimension of value. Defaults to 64.
            kv_stride (int, optional): stride of key and value downsampling. Defaults to 1.
            dw_kernel (int, optional): kernel size of key and value downsampling. Defaults to 3.
            enable_layer_scale (bool, optional): enable layer scale. Defaults to False.
        """
        super().__init__()
        self.num_head = num_head
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.norm = nn.BatchNorm2d(input_channel)
        self.query = nn.Conv2d(input_channel, num_head * key_dim, 1, bias=False)
        self.key = self.build_kv_projection(
            input_channel, key_dim, kv_stride, dw_kernel
        )
        self.value = self.build_kv_projection(
            input_channel, value_dim, kv_stride, dw_kernel
        )
        self.layers = nn.Sequential(
            nn.Conv2d(num_head * value_dim, input_channel, 1, bias=False),
        )

        if enable_layer_scale:
            self.layers.append(LayerScale2d(input_channel))

    def build_kv_projection(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
        kernel: int,
    ) -> nn.Module:
        layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, bias=False),
        )
        if stride > 1:
            layers.insert(
                0,
                Conv2dNormActivation(
                    input_channel,
                    input_channel,
                    kernel,
                    stride,
                    groups=input_channel,
                    activation_layer=None,
                ),
            )

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y = self.norm(x)

        # b, head, hw, key_dim
        q = self.query(y).view(b, self.num_head, self.key_dim, -1).transpose(-1, -2)
        # shared across heads, expand is a view without copy
        # b, head, h'w', key_dim
        k = (
            self.key(y)
            .flatten(2)
            .transpose(-1, -2)
            .unsqueeze(1)
            .expand(-1, self.num_head, -1, -1)
        )
        # b, head, h'w', value_dim
        v = (
            self.value(y)
            .flatten(2)
            .transpose(-1, -2)
            .unsqueeze(1)
            .expand(-1, self.num_head, -1, -1)
        )

        # b, head, hw, value_dim
        y = F.scaled_dot_product_attention(q, k, v)
        # b, head * value_dim, h, w
        y = y.transpose(-1, -2).reshape(b, self.num_head * self.value_dim, h, w)
        y = self.layers(y)

        return x + y
