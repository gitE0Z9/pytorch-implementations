import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models import FlattenFeature
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext
from torchlake.sequence_data.models.base.wrapper import (
    SequenceModelFullFeatureExtractor,
)


def pad_on_left(x: torch.Tensor, offset: int):
    return F.pad(x, (0, 0, offset, 0))


def pad_on_right(x: torch.Tensor, offset: int):
    return F.pad(x, (0, 0, 0, offset))


def shift_leftward(x: torch.Tensor, offset: int):
    return pad_on_right(x, offset)[:, offset:, :]


def shift_rightward(x: torch.Tensor, offset: int):
    return pad_on_left(x, offset)[:, :-offset, :]


class RCNN(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        context: NLPContext | None = None,
    ):
        """Recurrent convolution neural network in paper[9513-13-13041-1-2-20201228]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int): dimension of hidden layer
            output_size (int, optional): output size. Defaults to 1.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

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
        context, _ = self.foot(x)
        left_context, right_context = context.chunk(2, -1)
        # b, s, 2*h + e
        y = torch.cat(
            [
                shift_rightward(left_context, 1),
                embedded,
                shift_leftward(right_context, 1),
            ],
            -1,
        )
        # b, 2*h + e, s
        y = y.transpose(-1, -2)
        # b, h, s
        y = self.blocks(y).tanh()
        # b, h
        return self.head(y)
