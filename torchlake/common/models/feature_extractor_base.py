from abc import ABC, abstractmethod

from torch import nn
import torch
from torchvision.models._api import Weights


class ExtractorBase(nn.Module, ABC):
    def __init__(self, network_name: str, layer_type: str, trainable: bool = True):
        super(ExtractorBase, self).__init__()
        self.layer_type = layer_type
        self.trainable = trainable

        self.network_name = network_name
        self.weights: Weights = self.get_weight(network_name)
        self.feature_extractor = self.build_feature_extractor(
            self.network_name, self.weights
        )

    @abstractmethod
    def get_weight(self, network_name: str) -> Weights: ...

    @abstractmethod
    def build_feature_extractor(
        self,
        network_name: str,
        weights: Weights,
    ) -> nn.Module: ...

    @abstractmethod
    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: list[str],
    ) -> list[torch.Tensor]: ...
