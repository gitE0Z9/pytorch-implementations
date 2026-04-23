from math import prod
from typing import Sequence

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import FlattenFeature
from torchlake.common.models.model_base import ModelBase


def init_conv(layer: nn.Conv2d):
    nn.init.normal_(layer.weight, 0, 0.02)
    nn.init.normal_(layer.bias, 0.5, 0.02)


def init_dense(layer: nn.Linear):
    nn.init.normal_(layer.weight, 0, 0.2)
    if layer.bias is not None:
        nn.init.normal_(layer.bias, 0.5, 0.02)


class SiameseNet(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        image_shape: Sequence[int],
    ):
        self.hidden_dim = hidden_dim
        self.image_shape = image_shape
        super().__init__(input_channel, 1)

    @property
    def feature_dim(self) -> int:
        return 4096

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.hidden_dim,
                10,
                padding=0,
                norm_layer=None,
            ),
            nn.MaxPool2d(2, 2),
        )

        init_conv(self.foot[0][0])

    def build_blocks(self, **kwargs):
        n = prod(2 + (s - 73) // 8 if s > 72 else 1 for s in self.image_shape)
        self.blocks = nn.Sequential(
            Conv2dNormActivation(
                self.hidden_dim,
                self.hidden_dim * 2,
                7,
                padding=0,
                norm_layer=None,
            ),
            nn.MaxPool2d(2, 2),
            Conv2dNormActivation(
                self.hidden_dim * 2,
                self.hidden_dim * 2,
                4,
                padding=0,
                norm_layer=None,
            ),
            nn.MaxPool2d(2, 2),
            Conv2dNormActivation(
                self.hidden_dim * 2,
                self.hidden_dim * 4,
                4,
                padding=0,
                norm_layer=None,
            ),
            FlattenFeature(reduction=None),
            nn.Linear(n * self.hidden_dim * 4, self.feature_dim),
            nn.Sigmoid(),
        )
        init_conv(self.blocks[0][0])
        init_conv(self.blocks[2][0])
        init_conv(self.blocks[4][0])
        init_dense(self.blocks[6])

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, output_size, bias=False),
        )

        init_dense(self.head[0])

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y)
        return y

    def get_logit(
        self,
        query_vectors: torch.Tensor,
        support_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """get cross logits

        Args:
            query_vectors (torch.Tensor): shape (b, d)
            support_vectors (torch.Tensor): shape (b, d)

        Returns:
            torch.Tensor: shape (b, 1)
        """
        # b, 1
        return self.head((query_vectors - support_vectors).abs())

    def forward(
        self,
        query_set: torch.Tensor,
        support_set: torch.Tensor,
    ) -> torch.Tensor:
        query_vectors = self.feature_extract(query_set)
        support_vectors = self.feature_extract(support_set)

        # b, 1
        return self.get_logit(query_vectors, support_vectors)
