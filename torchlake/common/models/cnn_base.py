from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torchlake.common.models.flatten import FlattenFeature


class ModelBase(nn.Module, ABC):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
    ):
        super(ModelBase, self).__init__()
        self.build_foot(input_channel)
        self.build_blocks()
        self.build_head(output_size)

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    def config(self) -> list[list[Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_foot(self, input_channel: int):
        self.foot = ...

    @abstractmethod
    def build_blocks(self):
        self.blocks = ...

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            FlattenFeature(),
            nn.Linear(self.feature_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y)
        return self.head(y)
