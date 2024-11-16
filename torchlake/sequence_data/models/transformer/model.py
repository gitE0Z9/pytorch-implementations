import math

import torch
from torch import nn
from torchlake.common.models import PositionEncoding
from torchlake.common.models.model_base import ModelBase
from torchlake.common.utils.numerical import causal_mask

from .network import TransformerDecoderBlock, TransformerEncoderBlock


class TransformerEncoder(ModelBase):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        padding_idx: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        super().__init__(vocab_size, None)

    def build_foot(self, vocab_size):
        self.foot = nn.ModuleDict(
            {
                "pos_embed": PositionEncoding(trainable=False),
                "token_embed": nn.Embedding(
                    vocab_size,
                    self.hidden_dim,
                    padding_idx=self.padding_idx,
                ),
                "dropout": nn.Dropout(p=self.dropout_prob),
            }
        )

        # stated in paper p.5
        self.foot["token_embed"].weight.data.mul_(math.sqrt(self.hidden_dim))

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )

    def build_head(self, _):
        self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot["token_embed"](x)
        y = y + self.foot["pos_embed"](y).to(x.device)
        y = self.foot["dropout"](y)
        return self.blocks(y)


class TransformerDecoder(ModelBase):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        causal_mask: bool = True,
        padding_idx: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.causal_mask = causal_mask
        super().__init__(vocab_size, output_size)
        # remove flatten
        del self.head[0]

    @property
    def feature_dim(self) -> int:
        return self.hidden_dim

    def build_foot(self, vocab_size):
        self.foot = nn.ModuleDict(
            {
                "pos_embed": PositionEncoding(trainable=False),
                "token_embed": nn.Embedding(
                    vocab_size,
                    self.hidden_dim,
                    padding_idx=self.padding_idx,
                ),
                "dropout": nn.Dropout(p=self.dropout_prob),
            }
        )

        # stated in paper p.5
        self.foot["token_embed"].weight.data.mul_(math.sqrt(self.hidden_dim))

    def build_blocks(self):
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape

        y = self.foot["token_embed"](x)
        y = y + self.foot["pos_embed"](y).to(x.device)
        y = self.foot["dropout"](y)

        mask = None
        if self.causal_mask:
            mask = causal_mask(1, seq_len, seq_len)

        # encoder returned last layer representation
        # each layer of decoder receive same representation
        for block in self.blocks:
            y = block(y, encoded, mask)

        return self.head(y)


class Transformer(ModelBase):

    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
    ):
        super().__init__(
            foot_kwargs={"encoder": encoder},
            head_kwargs={"decoder": decoder},
        )

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("encoder")

    def build_head(self, _, **kwargs):
        self.head = kwargs.pop("decoder")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # encode
        z = self.foot(x)

        # decode
        return self.head(y, z)
