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
