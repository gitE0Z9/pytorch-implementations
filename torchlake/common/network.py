from typing import Literal

import torch
import torchvision
from torch import nn

from .constants import IMAGENET_MEAN, IMAGENET_STD
from .models.imagenet_norm import ImageNormalization


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
    ):
        super(ConvBnRelu, self).__init__()
        self.conv_last = conv_last

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
        self.bn = (
            nn.BatchNorm2d(output_channel if not conv_last else input_channel)
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
