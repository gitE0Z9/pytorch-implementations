import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext


class GloVe(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context: NLPContext | None = None,
    ):
        """GloVe: Global Vectors for Word Representation

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context = context
        super().__init__(vocab_size, 1)
        self.word_bias = nn.Parameter(torch.zeros((self.vocab_size,)))
        self.context_bias = nn.Parameter(torch.zeros((self.vocab_size,)))

    @property
    def input_embedding(self) -> nn.Embedding:
        return self.foot

    @property
    def output_embedding(self) -> nn.Embedding:
        return self.head

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Embedding(
            input_channel,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def build_head(self, _, **kwargs):
        self.head = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def forward(self, gram: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            gram (torch.Tensor): shape is batch_size*subseq_len, 1
            context (torch.Tensor): shape is batch_size*subseq_len, neighbor_size

        Returns:
            torch.Tensor: w_i * w_j + b_i + b_j
        """
        # b*seq, 1, h # b*seq, neighbor_size, h
        w_e, c_e = self.foot(gram), self.head(context)
        # b*seq, neighbor_size + b*seq, 1 + b*seq, neighbor_size => b*seq, neighbor_size
        return (
            torch.einsum("sxh,syh->sy", w_e, c_e)
            + self.word_bias[gram].masked_fill_(gram == self.context.padding_idx, 0)
            + self.context_bias[context].masked_fill_(
                context == self.context.padding_idx, 0
            )
        )
