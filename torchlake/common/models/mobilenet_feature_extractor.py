from typing import Literal, Sequence

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)

from torchlake.common.types import MOBILENET_NAMES

from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class MobileNetFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: MOBILENET_NAMES,
        layer_type: Literal["block"] = "block",
        trainable: bool = True,
    ):
        """mobilenet feature extractor

        Args:
            network_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]): torchvision mobilenet model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super().__init__(network_name, layer_type, trainable)
        self._feature_dim = self.feature_dims[-1][-1]
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[list[int]]:
        return {
            "mobilenet_v2": [
                [32, 16],
                [24] * 2,
                [32] * 3,
                [64] * 4 + [96] * 3,
                [160] * 3 + [320] + [1280],
            ],
            "mobilenet_v3_small": [
                [16],
                [16],
                [24],
                [40] * 3 + [48] * 2,
                [96] * 3 + [576],
            ],
            "mobilenet_v3_large": [
                [16, 16],
                [24] * 2,
                [40] * 3,
                [80] * 4 + [112] * 2,
                [160] * 3 + [960],
            ],
        }[self.network_name]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @feature_dim.setter
    def feature_dim(self, dim: int):
        self._feature_dim = dim

    def get_stage(self) -> list[list[int]]:
        return {
            "mobilenet_v2": [
                [0, 1],
                [2, 3],
                [4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18],
            ],
            "mobilenet_v3_small": [
                [0],
                [1],
                [2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            "mobilenet_v3_large": [
                [0, 1],
                [2, 3],
                [4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
        }[self.network_name]

    @property
    def hidden_dim_2x(self) -> list[int]:
        return self.feature_dims[0]

    @property
    def hidden_dim_4x(self) -> list[int]:
        return self.feature_dims[1]

    @property
    def hidden_dim_8x(self) -> list[int]:
        return self.feature_dims[2]

    @property
    def hidden_dim_16x(self) -> list[int]:
        return self.feature_dims[3]

    @property
    def hidden_dim_32x(self) -> list[int]:
        return self.feature_dims[4]

    def get_weight(self, network_name: str) -> Weights:
        return {
            "mobilenet_v2": MobileNet_V2_Weights.DEFAULT,
            "mobilenet_v3_small": MobileNet_V3_Small_Weights.DEFAULT,
            "mobilenet_v3_large": MobileNet_V3_Large_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: nn.Module = model_class(weights=weights).features

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: Sequence[
            Literal["0_1", "1_1", "2_1", "3_1", "4_1", "output"]
        ],
        normalization: bool = True,
    ) -> list[torch.Tensor]:
        if self.layer_type != "block":
            raise NotImplementedError

        targets = set(target_layer_names)

        features = []

        if normalization:
            img = self.normalization(img)

        y: torch.Tensor = img
        for stage_idx, stage in enumerate(self.get_stage()):
            for layer_count, layer_idx in enumerate(stage, start=1):
                layer = self.feature_extractor[layer_idx]
                y = layer(y)

                layer_name = f"{stage_idx}_{layer_count}"
                if layer_name in target_layer_names:
                    features.append(y)
                    targets.remove(layer_name)

                    if len(targets) == 0 and "output" not in targets:
                        break

        if "output" in targets:
            features.append(y.mean((2, 3)))

        return features
