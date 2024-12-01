from torchvision.models.efficientnet import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)


from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights

from ..types import EFFICIENTNET_V1_NAMES
from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class EfficientNetFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: EFFICIENTNET_V1_NAMES,
        layer_type: Literal["block"],
        trainable: bool = True,
    ):
        """resnet feature extractor

        Args:
            network_name (EFFICIENTNET_V1_NAMES): torchvision efficientnet model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        return {
            "b0": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
            "b1": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
            "b2": [32, 16, 24, 48, 88, 120, 208, 352, 1408],
            "b3": [40, 24, 32, 48, 96, 136, 232, 384, 1536],
            "b4": [48, 24, 32, 56, 112, 160, 272, 448, 1792],
            "b5": [48, 24, 40, 64, 128, 176, 304, 512, 2048],
            "b6": [56, 32, 40, 72, 144, 200, 344, 576, 2304],
            "b7": [64, 32, 48, 80, 160, 224, 384, 640, 2560],
        }[self.network_name]

    def get_weight(self, network_name: str) -> Weights:
        # https://github.com/pytorch/vision/issues/7744#issuecomment-1757321451
        return {
            "b0": EfficientNet_B0_Weights.DEFAULT,
            "b1": EfficientNet_B1_Weights.DEFAULT,
            "b2": EfficientNet_B2_Weights.DEFAULT,
            "b3": EfficientNet_B3_Weights.DEFAULT,
            "b4": EfficientNet_B4_Weights.DEFAULT,
            "b5": EfficientNet_B5_Weights.DEFAULT,
            "b6": EfficientNet_B6_Weights.DEFAULT,
            "b7": EfficientNet_B7_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, f"efficientnet_{network_name}")
        feature_extractor: nn.Module = model_class(weights=weights).features

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: Literal[
            "0_1",
            "1_1",
            "2_1",
            "3_1",
            "4_1",
            "5_1",
            "6_1",
            "7_1",
            "8_1",
            "output",
        ],
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
            features.append(y.mean((2, 3)).squeeze(-1))

        return features
