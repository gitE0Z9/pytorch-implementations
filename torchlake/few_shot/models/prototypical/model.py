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

        h = query_vectors.size(-1)
        query_vectors = query_vectors.view(n, q, h)
        support_vectors = support_vectors.view(n, k, h)

        prototypes = support_vectors.mean(1)

        # q, n, n
        return (
            torch.cdist(query_vectors.view(-1, h), prototypes)
            .reshape(n, q, n)
            .transpose(0, 1)
        )
