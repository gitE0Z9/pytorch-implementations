import torch
from torch import nn
from torchlake.common.models import FlattenFeature
from torchlake.common.schemas.nlp import NlpContext
from torchlake.sequence_data.models.base.wrapper import (
    SequenceModelFullFeatureExtractor,
)
from torchlake.common.models.model_base import ModelBase


class RCNN(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        context: NlpContext = NlpContext(),
    ):
        """Recurrent convolution neural network in paper[9513-13-13041-1-2-20201228]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int): dimension of hidden layer
            output_size (int, optional): output size. Defaults to 1.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context = context
        super().__init__(vocab_size, output_size)

    def build_foot(self, vocab_size: int):
        self.foot = SequenceModelFullFeatureExtractor(
            vocab_size,
            self.embed_dim,
            self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            context=self.context,
            model_class=nn.RNN,
        )

    def build_blocks(self):
        self.blocks = nn.Conv1d(
            2 * self.hidden_dim + self.embed_dim, self.hidden_dim, 1
        )

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            FlattenFeature(reduction="max", dimension="1d"),
            nn.Linear(self.hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, s, e
        embedded = self.foot.embed(x)
        # b, s, 2*h
        # XXX: computed twice
        context, _ = self.foot(x)
        left_context, right_context = context.chunk(2, -1)
        # b, s, 2*h + e
        y = torch.cat([left_context, embedded, right_context], -1)
        # b, 2*h + e, s
        y = y.transpose(-1, -2)
        # b, h, s
        y = self.blocks(y).tanh()
        # b, h
        return self.head(y)
