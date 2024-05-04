from typing import Literal

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from .constants import IMAGENET_MEAN, IMAGENET_STD


class SqueezeExcitation2d(nn.Module):

    def __init__(self, in_dim: int, reduction_ratio: float = 1):
        super(SqueezeExcitation2d, self).__init__()
        self.s = nn.Linear(in_dim, in_dim // reduction_ratio)
        self.e = nn.Linear(in_dim // reduction_ratio, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean((2, 3))
        y = self.s(y)
        y.relu_()
        y = self.e(y)
        return x * y.sigmoid()[:, :, None, None]


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
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            stride,
            padding,
            dilation,
            group,
            bias=not enable_bn,
        )
        self.bn = nn.BatchNorm2d(output_channel) if enable_bn else enable_bn
        self.activation = activation or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)

        return self.activation(y)


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


class ResBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        block: nn.Module,
        activation: nn.Module | None = nn.ReLU(True),
    ):
        """residual block
        skip connection is 1x1 conv shortcut if input_channel != output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            block (nn.Module): block class
            activation (tuple[nn.Module  |  None], optional): activation of residual output. Defaults to nn.ReLU(True).
        """
        super(ResBlock, self).__init__()
        self.activation = activation

        self.block = block

        self.downsample = (
            nn.Identity()
            if input_channel == output_channel
            else ConvBnRelu(input_channel, output_channel, 1, activation=None)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.downsample(x)

        if self.activation is None:
            return y
        else:
            return self.activation(y)


class ImageNormalization(nn.Module):
    def __init__(
        self,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ):
        super(ImageNormalization, self).__init__()
        ## C,1,1 shape for broadcasting
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, img: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        original_shape = img.size()

        if img.dim() == 3:
            img = img.unsqueeze(0)

        if not reverse:
            normalized = (img - self.mean.to(img.device)) / self.std.to(img.device)
        else:
            normalized = img * self.std.to(img.device) + self.mean.to(img.device)

        return normalized.reshape(*original_shape)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        network_name: str,
        layer_type: Literal["conv", "relu", "maxpool"],
        trainable: bool = True,
    ):
        super(FeatureExtractor, self).__init__()
        self.layer_type = layer_type
        self.trainable = trainable

        self.normalization = ImageNormalization(IMAGENET_MEAN, IMAGENET_STD)
        self.feature_extractor = self.build_feature_extractor(network_name)

    def build_feature_extractor(self, network_name: str) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor = model_class(weights="DEFAULT").features.eval()

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: list[str],
    ) -> list[torch.Tensor]:
        if self.layer_type == "conv":
            layer_class = nn.Conv2d
        elif self.layer_type == "relu":
            layer_class = nn.ReLU
        elif self.layer_type == "maxpool":
            layer_class = nn.MaxPool2d
        else:
            raise NotImplementedError

        features = []

        img = self.normalization(img)

        block_count = 1
        layer_count = 1
        for layer in self.feature_extractor:
            img = layer(img)

            layer_name = f"{block_count}_{layer_count}"
            if isinstance(layer, layer_class):
                if layer_name in target_layer_names:
                    features.append(img)
                layer_count += 1
            if isinstance(layer, nn.MaxPool2d):
                block_count += 1
                layer_count = 1

        return features
