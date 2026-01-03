from abc import ABC, abstractmethod
from functools import partial

from torch import nn
import torch
from torchvision.models._api import Weights


class ExtractorBase(nn.Module, ABC):
    def __init__(
        self,
        network_name: str,
        layer_type: str,
        trainable: bool = True,
    ):
        """feature extractor of image classifier, fully conncted layer will be deleted

        Args:
            network_name (str): name of network
            layer_type (str): which type of layer to focus on
            trainable (bool, optional): is extractor trainable. Defaults to True.
            enable_normalization (bool, optional):
        """
        super().__init__()
        self.layer_type = layer_type
        self.trainable = trainable

        self.network_name = network_name
        self.weights: Weights = self.get_weight(network_name)
        self.feature_extractor = self.build_feature_extractor(
            self.network_name, self.weights
        )

    @property
    def feature_dims(self) -> list[list[int]] | list[int]:
        raise NotImplementedError

    @property
    def hidden_dim_8x(self) -> int:
        raise NotImplementedError

    @property
    def hidden_dim_16x(self) -> int:
        raise NotImplementedError

    @property
    def hidden_dim_32x(self) -> int:
        raise NotImplementedError

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_weight(self, network_name: str) -> Weights:
        """get pytorch weight enum by network name

        Args:
            network_name (str): name of network

        Returns:
            Weights: pytorch weight enum
        """

    @abstractmethod
    def build_feature_extractor(
        self,
        network_name: str,
        weights: Weights,
    ) -> nn.Module:
        """build feature extractor by loading weight and model

        Args:
            network_name (str): name of network
            weights (Weights): pytorch weight enum

        Returns:
            nn.Module: model
        """

    def fix_target_layers(self, layer_names: list[str]):
        self.forward = partial(
            self.forward,
            target_layer_names=layer_names,
        )

    @abstractmethod
    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: list[str],
        normalization: bool = True,
    ) -> list[torch.Tensor]:
        """compute feature maps

        Args:
            img (torch.Tensor): image tensor
            target_layer_names (list[str]): which layers to extract
            normalization (bool, optional): normalize with imagenet statistics before feature extraction. Default to True.

        Returns:
            list[torch.Tensor]: extracted feature maps
        """
