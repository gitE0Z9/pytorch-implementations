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
        )
        init_conv(self.blocks[0][0])
        init_conv(self.blocks[2][0])
        init_conv(self.blocks[4][0])

    def build_neck(self, **kwargs):
        n = prod(2 + (s - 73) // 8 if s > 72 else 1 for s in self.image_shape)
        self.neck = nn.Sequential(
            FlattenFeature(reduction=None),
            nn.Linear(n * self.hidden_dim * 4, self.feature_dim),
        )

        init_dense(self.neck[1])

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, output_size, bias=False),
        )

        init_dense(self.head[0])

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y)
        return self.neck(y)

    def get_logit(
        self,
        query_vectors: torch.Tensor,
        support_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """get cross logits

        Args:
            query_vectors (torch.Tensor): shape (n1, d)
            support_vectors (torch.Tensor): shape (n2, d)

        Returns:
            torch.Tensor: shape (n1, n2)
        """
        d = query_vectors.size(-1)
        query_vectors = query_vectors.view(-1, 1, d)
        support_vectors = support_vectors.view(1, -1, d)

        # n1, n2
        return self.head((query_vectors - support_vectors).abs()).squeeze(-1)

    def forward(
        self,
        query_set: torch.Tensor,
        support_set: torch.Tensor,
    ) -> torch.Tensor:
        # q is query size
        q = query_set.size(1)
        # n is n way
        # k is k shot
        n, k, c, h, w = support_set.shape
        assert k == 1, "only one shot supported"

        query_vectors = self.feature_extract(query_set.view(-1, c, h, w))
        support_vectors = self.feature_extract(support_set.view(-1, c, h, w))

        y = (
            # n*q, n*k => n, q, n, k
            self.get_logit(query_vectors, support_vectors).reshape(n, q, n, k)
            # q, n, n, k
            .permute(1, 0, 2, 3)
            # q, n, n
            .squeeze(-1)
        )

        return y
