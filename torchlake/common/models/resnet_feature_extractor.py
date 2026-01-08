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

from torchlake.common.models.flatten import FlattenFeature

from ..types import RESNET_NAMES
from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class ResNetFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: RESNET_NAMES,
        layer_type: Literal["block"] = "block",
        trainable: bool = True,
        enable_gap: bool = True,
    ):
        """resnet feature extractor

        Args:
            network_name (Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]): torchvision resnet model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        self.enable_gap = enable_gap
        super().__init__(network_name, layer_type, trainable)
        self._feature_dim = self.feature_dims[-1]
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        # legacy: "1_1", "2_1", "3_1", "4_1", "output"
        return {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "resnet152": [256, 512, 1024, 2048],
        }[self.network_name]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @feature_dim.setter
    def feature_dim(self, dim: int):
        self._feature_dim = dim

    @property
    def hidden_dim_stem(self) -> int:
        return {
            "resnet18": 64,
            "resnet34": 64,
            "resnet50": 64,
            "resnet101": 64,
            "resnet152": 64,
        }[self.network_name]

    @property
    def hidden_dim_4x(self) -> int:
        return {
            "resnet18": 64,
            "resnet34": 64,
            "resnet50": 256,
            "resnet101": 256,
            "resnet152": 256,
        }[self.network_name]

    @property
    def hidden_dim_8x(self) -> int:
        return {
            "resnet18": 128,
            "resnet34": 128,
            "resnet50": 512,
            "resnet101": 512,
            "resnet152": 512,
        }[self.network_name]

    @property
    def hidden_dim_16x(self) -> int:
        return {
            "resnet18": 256,
            "resnet34": 256,
            "resnet50": 1024,
            "resnet101": 1024,
            "resnet152": 1024,
        }[self.network_name]

    @property
    def hidden_dim_32x(self) -> int:
        return {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048,
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
        model: ResNet = model_class(weights=weights)

        fe = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        if self.enable_gap:
            fe.append(FlattenFeature())

        if not self.trainable:
            for param in fe.parameters():
                param.requires_grad = False

        return fe

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: Literal["0_1", "1_1", "2_1", "3_1", "4_1", "output"],
    ) -> list[torch.Tensor]:
        if self.layer_type != "block":
            raise NotImplementedError

        targets = {target_layer_names}

        features = []

        img = self.normalization(img)

        y = self.feature_extractor[:4](img)
        if "0_1" in targets:
            features.append(y)

        for i in range(4, 8):
            layer_name = f"{i-3}_1"
            y = self.feature_extractor[i](y)
            if layer_name in targets:
                features.append(y)
                targets.remove(layer_name)

                if len(targets) == 0:
                    break

        if "output" in targets:
            y = self.feature_extractor[-1](y)
            features.append(y)

        return features
