from math import prod
from typing import Sequence

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase


class RelationNet(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int = 64,
        num_layer: int = 4,
        image_shape: Sequence[int] = (32, 32),
    ):
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.image_shape = image_shape
        super().__init__(input_channel, 1)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, self.hidden_dim, 3),
            nn.MaxPool2d(2),
            Conv2dNormActivation(self.hidden_dim, self.hidden_dim, 3),
            nn.MaxPool2d(2),
        )

    def build_blocks(self, **kwargs):
        blocks = []
        for _ in range(self.num_layer - 2):
            blocks.append(
                Conv2dNormActivation(self.hidden_dim, self.hidden_dim, 3),
            )

        self.blocks = nn.Sequential(*blocks)

    def build_neck(self, **kwars):
        self.neck = nn.Sequential(
            Conv2dNormActivation(2 * self.hidden_dim, self.hidden_dim, 3),
            nn.MaxPool2d(2),
            Conv2dNormActivation(self.hidden_dim, self.hidden_dim, 3),
            nn.MaxPool2d(2),
            nn.Flatten(1, -1),
        )

    def build_head(self, output_size, **kwargs):
        d = prod(s // 16 for s in self.image_shape) * self.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(d, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

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
            query_vectors (torch.Tensor): shape (n*q or q, c, h // 4, w // 4)
            support_vectors (torch.Tensor): shape (n, c, h // 4, w // 4)

        Returns:
            torch.Tensor: shape (n*q, n)
        """
        n = support_vectors.size(0)
        query_size = query_vectors.size(0)

        # n*q, n, c, h, w
        query_vectors = query_vectors.unsqueeze(1).repeat(1, n, 1, 1, 1)
        # n*q, n, c, h, w
        support_vectors = support_vectors.unsqueeze(0).repeat(query_size, 1, 1, 1, 1)

        # n*q, n, 2c, h, w
        y = torch.cat((query_vectors, support_vectors), 2)
        # n*q*n, 2c, h, w => n*q*n, d
        y = self.neck(y.view(-1, *y.shape[2:]))
        # n*q, n
        return self.head(y).view(query_size, n)

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

        query_vectors = self.feature_extract(query_set.view(-1, c, h, w))
        support_vectors = self.feature_extract(support_set.view(-1, c, h, w))

        # n, c, h, w
        support_vectors = support_vectors.view(n, k, *support_vectors.shape[1:]).sum(1)

        # q, ?, n
        return (
            self.get_logit(query_vectors, support_vectors)
            .reshape(-1, q, n)
            .transpose(0, 1)
        )
