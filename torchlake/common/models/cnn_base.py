from abc import ABC, abstractmethod

import torch
from torch import nn
from torchlake.common.models.flatten import FlattenFeature


class ModelBase(ABC):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
    ):
        self.feature_dim = ...

        self.build_foot(input_channel)
        self.build_blocks()
        self.build_head(output_size)

    @abstractmethod
    def build_foot(self, input_channel: int): ...

    @abstractmethod
    def build_blocks(self): ...

    def build_head(self, output_size: int):
        self.pool = FlattenFeature()
        self.fc = nn.Linear(self.feature_dim, output_size)

        self.head = nn.Sequential(
            self.pool,
            self.fc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y)
        return self.head(y)
