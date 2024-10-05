from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models.resnet import (
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNet,
)

from .imagenet_normalization import ImageNetNormalization
from torchvision.models._api import Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        network_name: Literal["resnet50", "resnet101", "resnet152"],
        layer_type: Literal["maxpool"],
        trainable: bool = True,
        drop_fc: bool = True,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.trainable = trainable
        self.drop_fc = drop_fc

        self.normalization = ImageNetNormalization()
        self.weights: Weights = self.get_weight(network_name)
        self.feature_extractor: ResNet = self.build_feature_extractor(
            network_name, self.weights
        )

    def get_weight(self, network_name: str) -> Weights:
        return {
            "resnet50": ResNet50_Weights.DEFAULT,
            "resnet101": ResNet101_Weights.DEFAULT,
            "resnet152": ResNet152_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: ResNet = model_class(weights=weights).eval()

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        if self.drop_fc:
            del feature_extractor.fc

        # fuse head
        feature_extractor.add_module(
            "head",
            nn.Sequential(
                feature_extractor.conv1,
                feature_extractor.bn1,
                feature_extractor.relu,
                feature_extractor.maxpool,
            ),
        )
        # del feature_extractor.conv1
        # del feature_extractor.bn1
        # del feature_extractor.relu
        # del feature_extractor.maxpool
        feature_extractor = nn.Sequential(
            feature_extractor.head,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool,
        )

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: list[str],
    ) -> list[torch.Tensor]:
        if not self.layer_type == "maxpool":
            raise NotImplementedError

        features = []

        img = self.normalization(img)

        y = img
        for i, layer in enumerate(self.feature_extractor):
            y = layer(y)
            if f"{i}_1" in target_layer_names:
                features.append(y)

        if "output" in target_layer_names:
            features.append(y.squeeze_(2, 3))

        return features
