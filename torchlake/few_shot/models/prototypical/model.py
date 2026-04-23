import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase


class PrototypicalNet(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int = 64,
        num_layer: int = 4,
    ):
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        super().__init__(input_channel, 1)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, self.hidden_dim, 3),
            nn.MaxPool2d(2),
        )

    def build_blocks(self, **kwargs):
        blocks = []
        for _ in range(self.num_layer - 1):
            blocks.extend(
                (
                    Conv2dNormActivation(self.hidden_dim, self.hidden_dim, 3),
                    nn.MaxPool2d(2),
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def build_head(self, output_size, **kwargs):
        self.head = lambda x: x.flatten(1, -1)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y)
        y = self.head(y)
        return y

    def get_logit(
        self,
        query_vectors: torch.Tensor,
        support_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """get cross logits

        Args:
            query_vectors (torch.Tensor): shape (n*q or q, d)
            support_vectors (torch.Tensor): shape (n, d)

        Returns:
            torch.Tensor: shape (n*q, n)
        """
        return torch.cdist(query_vectors, support_vectors)

    def forward(
        self,
        query_set: torch.Tensor,
        support_set: torch.Tensor,
    ) -> torch.Tensor:
        """forward

        Args:
            query_set (torch.Tensor): shape (n, q, c, h, w)
            support_set (torch.Tensor): shape (n, k, c, h, w)

        Returns:
            torch.Tensor: shape (q, n, n)
        """
        # q is query size
        q = query_set.size(1)
        # n is n way
        # k is k shot
        n, k, c, h, w = support_set.shape

        # n*q, d
        query_vectors = self.feature_extract(query_set.view(-1, c, h, w))
        # n*k, d
        support_vectors = self.feature_extract(support_set.view(-1, c, h, w))

        d = query_vectors.size(-1)
        # n, k, d
        support_vectors = support_vectors.view(n, k, d)

        # n, d
        prototypes = support_vectors.mean(1)

        # q, n, n
        return (
            self.get_logit(query_vectors, prototypes).reshape(-1, q, n).transpose(0, 1)
        )
