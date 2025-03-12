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
        foot_kwargs: dict | None = {},
        blocks_kwargs: dict | None = {},
        neck_kwargs: dict | None = {},
        head_kwargs: dict | None = {},
    ):
        super().__init__()
        self.build_foot(input_channel, **foot_kwargs)
        self.build_blocks(**blocks_kwargs)
        self.build_neck(**neck_kwargs)
        self.build_head(output_size, **head_kwargs)

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    def config(self) -> list[list[Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_foot(self, input_channel: int, **kwargs):
        self.foot = ...

    def build_blocks(self, **kwargs):
        self.blocks = None

    def build_neck(self, **kwargs):
        self.neck = None

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            FlattenFeature(),
            nn.Linear(self.feature_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)

        if self.blocks is not None:
            y = self.blocks(y)

        if self.neck is not None:
            y = self.neck(y)

        return self.head(y)
