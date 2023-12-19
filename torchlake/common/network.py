from typing import Literal

import torch
import torchvision
from torch import nn

from .constants import IMAGENET_MEAN, IMAGENET_STD


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
        enable_relu: bool = True,
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
        self.relu = nn.ReLU(True) if enable_relu else enable_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        if self.relu:
            y = self.relu(y)

        return y


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
    ):
        super(FeatureExtractor, self).__init__()
        self.layer_type = layer_type

        self.normalization = ImageNormalization(IMAGENET_MEAN, IMAGENET_STD)
        self.feature_extractor = self.build_feature_extractor(network_name)

    def build_feature_extractor(self, network_name: str) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        return model_class(weights="DEFAULT").features.eval()

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
