import torch
import torchvision
from torch import nn
from typing import Literal
from torchlake.common.network import ImageNormalization
from torchlake.common.constants import IMAGENET_MEAN, IMAGENET_STD


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        network_name: str,
        layer_type: Literal["conv", "relu"],
        device: str,
    ):
        super(FeatureExtractor, self).__init__()
        self.layer_type = layer_type
        self.device = device

        self.normalization = ImageNormalization(IMAGENET_MEAN, IMAGENET_STD)
        self.feature_extractor = self.build_feature_extractor(network_name)

    def build_feature_extractor(self, network_name: str) -> nn.Module:
        return (
            getattr(torchvision.models, network_name)(weights="DEFAULT")
            .features.to(self.device)
            .eval()
        )

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: list[str],
    ) -> list[torch.Tensor]:
        if self.layer_type == "conv":
            layer_class = nn.Conv2d
        elif self.layer_type == "relu":
            layer_class = nn.ReLU
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
            elif isinstance(layer, nn.MaxPool2d):
                block_count += 1
                layer_count = 1

        return features
