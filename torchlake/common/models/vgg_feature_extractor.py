from typing import Literal

import torch
import torchvision
from torch import nn

from .imagenet_normalization import ImageNetNormalization
from torchvision.models.vgg import VGG16_Weights


class VggFeatureExtractor(nn.Module):
    def __init__(
        self,
        network_name: str,
        layer_type: Literal["conv", "relu", "maxpool"],
        trainable: bool = True,
    ):
        super(VggFeatureExtractor, self).__init__()
        self.layer_type = layer_type
        self.trainable = trainable

        self.normalization = ImageNetNormalization()
        self.feature_extractor = self.build_feature_extractor(network_name)

    def build_feature_extractor(self, network_name: str) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: nn.Module = model_class(
            weights=VGG16_Weights.DEFAULT
        ).features.eval()

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
