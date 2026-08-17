import torch
import torch.nn.functional as F
from torch import nn

from .network import GCNLayer, GCNResBlock

from torchlake.common.models.model_base import ModelBase


class GCN(ModelBase):
    def __init__(self, input_channel: int, hidden_dim: int, output_size: int):
        self.hidden_dim = hidden_dim
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = GCNLayer(input_channel, self.hidden_dim)

    def build_head(self, output_size, **kwargs):
        self.head = GCNLayer(self.hidden_dim, output_size)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.foot(x, a)
        return self.head(y, a)


class GCNResidual(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_size: int,
        num_block: int = 3,
    ):
        self.hidden_dim = hidden_dim
        self.num_block = num_block
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = GCNLayer(input_channel, self.hidden_dim)

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                GCNResBlock(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_block)
            ]
        )

    def build_head(self, output_size, **kwargs):
        self.head = GCNLayer(self.hidden_dim, output_size)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.foot(x, a)
        for layer in self.blocks:
            y = F.relu(y, True)
            y = layer(y, a)

        return self.head(y, a)
