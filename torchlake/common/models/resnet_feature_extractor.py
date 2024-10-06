from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.resnet import (
    ResNet,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class ResNetFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: Literal["resnet50", "resnet101", "resnet152"],
        layer_type: Literal["block"],
        trainable: bool = True,
        drop_fc: bool = True,
    ):
        self.drop_fc = drop_fc
        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    def get_weight(self, network_name: str) -> Weights:
        return {
            "resnet50": ResNet50_Weights.DEFAULT,
            "resnet101": ResNet101_Weights.DEFAULT,
            "resnet152": ResNet152_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: ResNet = model_class(weights=weights)  # .eval()

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        if self.drop_fc:
            del feature_extractor.fc

        # fuse foot
        feature_extractor.add_module(
            "foot",
            nn.Sequential(
                feature_extractor.conv1,
                feature_extractor.bn1,
                feature_extractor.relu,
                feature_extractor.maxpool,
            ),
        )

        feature_extractor = nn.Sequential(
            feature_extractor.foot,
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
        target_layer_names: Literal["0_1", "1_1", "2_1", "3_1", "4_1", "output"],
    ) -> list[torch.Tensor]:
        if self.layer_type != "block":
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
