from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.resnet import (
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from ..types import RESNET_NAMES
from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class ResNetFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: RESNET_NAMES,
        layer_type: Literal["block"],
        trainable: bool = True,
    ):
        """resnet feature extractor

        Args:
            network_name (Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]): torchvision resnet model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        return {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "resnet152": [256, 512, 1024, 2048],
        }[self.network_name]

    def get_weight(self, network_name: str) -> Weights:
        return {
            "resnet18": ResNet18_Weights.DEFAULT,
            "resnet34": ResNet34_Weights.DEFAULT,
            "resnet50": ResNet50_Weights.DEFAULT,
            "resnet101": ResNet101_Weights.DEFAULT,
            "resnet152": ResNet152_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: ResNet = model_class(weights=weights)

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

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
