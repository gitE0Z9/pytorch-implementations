import torch
from torch import nn

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.models.model_base import ModelBase


class VLBL(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        neighbor_size: int,
        context: NLPContext | None = None,
    ):
        """vector log bilinear model in paper [Learning word embeddings efficiently with noise-contrastive estimation]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            neighbor_size (int): how many tokens around the gram, i.e. context size - 1. Defaults to 1.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neighbor_size = neighbor_size
        self.context = context
        super().__init__(vocab_size, 1)
        self.bias = nn.Parameter(torch.zeros((vocab_size,)))

    @property
    def input_embedding(self) -> nn.Embedding:
        return self.foot

    @property
    def output_embedding(self) -> nn.Embedding:
        return self.head

    def build_foot(self, _, **kwargs):
        self.foot = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def build_neck(self, **kwargs):
        # page 3
        # context_position_weights can also be average, like a topic model
        self.neck = nn.Conv1d(self.neighbor_size, 1, 1, bias=False)

    def build_head(self, _, **kwargs):
        self.head = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def forward(self, context: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
        """vLBL forward

        Args:
            context (torch.Tensor): surrounding words, shape is (batch_size*#subsequence, neighbor_size)
            gram (torch.Tensor): center word, shape is (batch_size*#subsequence, 1)

        Returns:
            torch.Tensor: similarity from context to gram
        """
        # b*s, c-1, h # b*s, 1, h
        c_e, w_e = self.foot(context), self.head(gram)
        # b*s, 1, h
        c_e = self.neck(c_e)
        # (b*s, 1, h x b*s, 1, h) => b*s, 1
        # b*s, 1 + b*s, 1 => b*s, 1
        return torch.einsum("nxh,nxh->nx", w_e, c_e) + self.bias[gram].masked_fill_(
            gram == self.context.padding_idx, 0
        )


class IVLBL(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        neighbor_size: int,
        context: NLPContext | None = None,
    ):
        """inverse vector log bilinear model in paper [Learning word embeddings efficiently with noise-contrastive estimation]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            neighbor_size (int): how many tokens around the gram, i.e. context size - 1. Defaults to 1.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neighbor_size = neighbor_size
        self.context = context
        super().__init__(vocab_size, 1)
        self.bias = nn.Parameter(torch.zeros((vocab_size,)))

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

    def build_neck(self, **kwargs):
        # page 3
        # context_position_weights can also be average, like a topic model
        self.neck = nn.Parameter(torch.ones((self.neighbor_size,)))

    def build_head(self, _, **kwargs):
        self.head = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def forward(self, gram: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """ivLBL forward

        Args:
            gram (torch.Tensor): center word, shape is (batch_size*#subsequence, 1)
            context (torch.Tensor): surrounding words, shape is (batch_size*#subsequence, neighbor_size)

        Returns:
            torch.Tensor: similarity from gram to context
        """
        # b*s, 1, h / b*s, c-1, h
        w_e, c_e = self.foot(gram), self.head(context)
        # b*s, 1, h * 1, c-1, 1 => b*s, c-1, h
        w_e = w_e * self.neck.view(1, self.neighbor_size, 1)
        # (b*s, c-1, h x b*s, c-1, h) => b*s, c-1
        # b*s, c-1 + b*s, c-1 => b*s, c-1
        return torch.einsum("nxh,nxh->nx", w_e, c_e) + self.bias[context].masked_fill_(
            context == self.context.padding_idx, 0
        )
