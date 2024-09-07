import torch
from torch import nn
from torchlake.common.models import FlattenFeature, KmaxPool1d
from torchlake.common.models.cnn_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext

from .network import Block, Folding, WideConv1d


class Dcnn(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 48,
        output_size: int = 1,
        kernels: tuple[int] = (7, 5),
        hidden_dims: tuple[int] = (6, 14),
        topk: int = 4,
        context: NlpContext = NlpContext(),
    ):
        assert len(hidden_dims) == len(
            kernels
        ), "kernels and hidden_dims should have same lengths."

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.kernels = kernels
        self.hidden_dims = hidden_dims
        self.topk = topk
        self.context = context
        super(Dcnn, self).__init__(1, output_size)

    @property
    def feature_dim(self) -> int:
        return self.topk * self.hidden_dims[-1] // 2

    def build_foot(self, input_channel: int):
        self.foot = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            self.context.padding_idx,
        )

    def build_blocks(self):
        num_conv = len(self.kernels)

        self.blocks = nn.Sequential(
            *[
                Block(
                    self.embed_dim if i == 0 else self.hidden_dims[i - 1],
                    hidden_dim,
                    kernel,
                    self.topk,
                    self.context.max_seq_len,
                    i + 1,
                    num_conv,
                )
                for i, (hidden_dim, kernel) in enumerate(
                    zip(
                        self.hidden_dims[:-1],
                        self.kernels[:-1],
                    )
                )
            ],
            Folding(),
            KmaxPool1d(self.topk),
        )

        self.blocks.insert(
            num_conv - 1,
            WideConv1d(
                self.hidden_dims[-2] if num_conv >= 2 else self.embed_dim,
                self.hidden_dims[-1],
                self.kernels[-1],
            ),
        )

        # penultimate layer has a dropout
        if num_conv >= 2:
            self.blocks.insert(
                num_conv - 1,
                nn.Dropout(),
            )

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            FlattenFeature(None, "1d"),
            nn.Linear(self.feature_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x).transpose(-1, -2)
        y = self.blocks(y)
        return self.head(y)
