from typing import Literal, Sequence

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.vgg import (
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
    VGG11_BN_Weights,
    VGG13_BN_Weights,
    VGG16_BN_Weights,
    VGG19_BN_Weights,
    VGG,
)

from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class VGGFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
        layer_type: Literal["conv", "relu", "maxpool", "gap"],
        trainable: bool = True,
        enable_bn: bool = False,
        enable_gap: bool = False,
        enable_fc1: bool = False,
        enable_fc2: bool = False,
        convert_fc_to_conv: bool = False,
        return_pooling_indices: bool = False,
    ):
        """VGG feature extractor

        Args:
            network_name (Literal["vgg11", "vgg13", "vgg16", "vgg19"]): torchvision vgg model
            layer_type (Literal["conv", "relu", "maxpool"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        self.enable_bn = enable_bn
        self.enable_gap = enable_gap
        self.enable_fc1 = enable_fc1
        self.enable_fc2 = enable_fc2
        self.convert_fc_to_conv = convert_fc_to_conv
        self.return_pooling_indices = return_pooling_indices
        self._feature_dim = 4096

        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        return [64, 128, 256, 512, 512]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @feature_dim.setter
    def feature_dim(self, dim: int):
        self._feature_dim = dim

    def get_stage(self) -> Sequence[Sequence[int]]:
        return {
            "vgg11": (
                (0,),
                (1,),
                (2, 3),
                (4, 5),
                (6, 7),
            ),
            "vgg13": (
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
            ),
            "vgg16": (
                (0, 1),
                (2, 3),
                (4, 5, 6),
                (7, 8, 9),
                (10, 11, 12),
            ),
            "vgg19": (
                (0, 1),
                (2, 3),
                (4, 5, 6, 7),
                (8, 9, 10, 11),
                (12, 13, 14, 15),
            ),
        }[self.network_name]

    @property
    def hidden_dim_2x(self) -> int:
        return 64

    @property
    def hidden_dim_4x(self) -> int:
        return 128

    @property
    def hidden_dim_8x(self) -> int:
        return 256

    @property
    def hidden_dim_16x(self) -> int:
        return 512

    @property
    def hidden_dim_32x(self) -> int:
        return 512

    def get_weight(self, network_name: str) -> Weights:
        return {
            "vgg11": VGG11_Weights.DEFAULT if not self.enable_bn else VGG11_BN_Weights,
            "vgg13": VGG13_Weights.DEFAULT if not self.enable_bn else VGG13_BN_Weights,
            "vgg16": VGG16_Weights.DEFAULT if not self.enable_bn else VGG16_BN_Weights,
            "vgg19": VGG19_Weights.DEFAULT if not self.enable_bn else VGG19_BN_Weights,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        module_name = network_name
        if self.enable_bn:
            module_name += "_bn"
        model_class = getattr(torchvision.models, module_name)
        m: VGG = model_class(weights=weights)

        fe: nn.Sequential = m.features

        if self.enable_gap:
            fe.append(m.avgpool)

        if self.enable_fc1:
            if self.convert_fc_to_conv:
                l = nn.Conv2d(512, 4096, 7)
                l.weight.data.copy_(m.classifier[0].weight.data.view(4096, 512, 7, 7))
                l.bias.data.copy_(m.classifier[0].bias.data)
                fe.append(l)
            else:
                fe.append(m.classifier[0])
            # relu
            fe.append(m.classifier[1])

        if self.enable_fc2:
            if self.convert_fc_to_conv:
                l = nn.Conv2d(4096, 4096, 1)
                l.weight.data.copy_(m.classifier[3].weight.data.view(4096, 4096, 1, 1))
                l.bias.data.copy_(m.classifier[3].bias.data)
                fe.append(l)
            else:
                fe.append(m.classifier[3])
            # relu
            fe.append(m.classifier[4])

        if self.return_pooling_indices:
            for _, m in fe.named_modules():
                if isinstance(m, nn.MaxPool2d):
                    m.return_indices = True

        if not self.trainable:
            for param in fe.parameters():
                param.requires_grad = False

        return fe

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: Sequence[str],
        normalization: bool = True,
    ) -> list[torch.Tensor]:
        if self.layer_type == "conv":
            check_function = lambda layer: isinstance(layer, nn.Conv2d)
        elif self.layer_type == "relu":
            check_function = lambda layer: isinstance(layer, nn.ReLU)
        elif self.layer_type == "maxpool":
            check_function = lambda layer: isinstance(layer, nn.MaxPool2d)
        elif self.layer_type == "gap":
            check_function = lambda layer: isinstance(layer, nn.AdaptiveAvgPool2d)
        else:
            raise NotImplementedError

        targets = set(target_layer_names)

        features = []
        pooling_indices = []

        if normalization:
            img = self.normalization(img)

        y = img
        stage_count = 1
        layer_count = 1
        for layer in self.feature_extractor:
            y = layer(y)

            if isinstance(layer, nn.MaxPool2d) and self.return_pooling_indices:
                y, pooling_index = y
                pooling_indices.append(pooling_index)

            layer_name = f"{stage_count}_{layer_count}"
            if check_function(layer):
                layer_count += 1

                if layer_name in targets:
                    features.append(y)
                    targets.remove(layer_name)

                    if len(targets) == 0:
                        break

            if isinstance(layer, nn.MaxPool2d | nn.AvgPool2d):
                stage_count += 1
                layer_count = 1

        if self.return_pooling_indices:
            return features, pooling_indices

        return features
