from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.vgg import (
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
)

from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class VGGFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
        layer_type: Literal["conv", "relu", "maxpool"],
        trainable: bool = True,
    ):
        """vgg feature extractor

        Args:
            network_name (Literal["vgg11", "vgg13", "vgg16", "vgg19"]): torchvision vgg model
            layer_type (Literal["conv", "relu", "maxpool"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super(VGGFeatureExtractor, self).__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        return [64, 128, 256, 512, 512]

    def get_weight(self, network_name: str) -> Weights:
        return {
            "vgg11": VGG11_Weights.DEFAULT,
            "vgg13": VGG13_Weights.DEFAULT,
            "vgg16": VGG16_Weights.DEFAULT,
            "vgg19": VGG19_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: nn.Module = model_class(weights=weights).features.eval()

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

        y = img
        stage_count = 1
        layer_count = 1
        for layer in self.feature_extractor:
            y = layer(y)

            layer_name = f"{stage_count}_{layer_count}"
            if isinstance(layer, layer_class):
                if layer_name in target_layer_names:
                    features.append(y)
                layer_count += 1
            if isinstance(layer, nn.MaxPool2d):
                stage_count += 1
                layer_count = 1

        return features
