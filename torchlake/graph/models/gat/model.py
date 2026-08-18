import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase

from .network import Block


class GAT(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_size: int = 1,
        num_heads: int = 4,
        num_block: int = 1,
        dropout_prob: float = 0.6,
        version: 1 | 2 = 1,
    ):
        """Graph attention network [1710.10903v3]

        Args:
            input_channel (int): input dimension
            hidden_dim (int): hidden dimension
            output_size (int): output dimension. Defaults to 1.
            num_heads (int, optional): number of heads of multi-head-attention. Defaults to 4.
            num_block (int, optional): number of blocks. Defaults to 1.
            dropout_prob (float, optional): dropout probability. Defaults to 0.6.
            version (int, optional): use v1 or v2. Defaults to 1.
        """
        assert version in (1, 2), "Layer version not supported."

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_block = num_block
        self.dropout_prob = dropout_prob
        self.version = version
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = Block(
            input_channel,
            self.hidden_dim,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
            version=self.version,
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.num_heads * self.hidden_dim,
                    self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_prob=self.dropout_prob,
                    version=self.version,
                )
                for _ in range(self.num_block)
            ],
        )

    def build_head(self, output_size, **kwargs):
        self.head = Block(
            self.num_heads * self.hidden_dim,
            output_size,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
            version=self.version,
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, input_channel)
            edges (torch.Tensor): edges, shape is (#edge, 2)

        Returns:
            torch.Tensor: output tensor, shape is (#node, output_size)
        """
        y = self.foot(x, edges, predict=False)
        for layer in self.blocks:
            y = layer(y, edges, predict=False)

        return self.head(y, edges, predict=True)
