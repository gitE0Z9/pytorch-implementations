import math

import torch
from torch import nn
from torchlake.common.models import PositionEncoding
from torchlake.common.utils.numerical import causal_mask

from .network import TransformerDecoderBlock, TransformerEncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        padding_idx: int | None = None,
    ):
        # for mark
        self.hidden_dim = hidden_dim
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=padding_idx,
        )
        # stated in paper p.5
        self.token_embedding.weight.data.mul_(math.sqrt(hidden_dim))

        self.position_encoding = PositionEncoding(trainable=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(hidden_dim, num_heads, dropout_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_embedding(x)
        y = y + self.position_encoding(y)
        y = self.dropout(y)
        y = self.blocks(y)

        return y


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        padding_idx: int | None = None,
    ):
        # for mark
        self.hidden_dim = hidden_dim
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=padding_idx,
        )
        # stated in paper p.5
        self.token_embedding.weight.data.mul_(math.sqrt(hidden_dim))

        self.position_encoding = PositionEncoding(trainable=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_dim, num_heads, dropout_prob)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape

        y = self.token_embedding(x)
        y = y + self.position_encoding(y)
        y = self.dropout(y)

        # encoder returned last layer representation
        # each layer of decoder receive same representation
        mask = causal_mask(1, seq_len, seq_len)
        for block in self.blocks:
            y = block(y, encoded, mask)

        return self.fc(y)
