import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.language_model.constants.enum import LossType
from torchlake.language_model.models.subword.network import SubwordEmbedding
from torchlake.common.models.model_base import ModelBase


class FastText(ModelBase):
    def __init__(
        self,
        bucket_size: int,
        embed_dim: int,
        output_size: int,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        context: NlpContext = NlpContext(),
    ):
        """FastText [1607.04606]

        Args:
            bucket_size (int): size of hash bucket
            embed_dim (int): embedding dimension
            output_size (int, optional): number of features of output. Defaults to 1.
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CE.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        self.embed_dim = embed_dim
        self.loss_type = loss_type
        self.context = context
        super().__init__(bucket_size, output_size)

    @property
    def embeddings(self) -> nn.Embedding:
        return self.foot.embeddings

    def build_foot(self, bucket_size: int):
        self.foot = SubwordEmbedding(
            bucket_size,
            self.embed_dim,
            context=self.context,
        )

    def build_blocks(self):
        self.blocks = ...

    def build_head(self, output_size: int):
        if self.loss_type == LossType.CROSS_ENTROPY:
            self.head = nn.Linear(self.embed_dim, output_size)
        else:
            self.head = nn.Identity()

    def get_sentence_vector(
        self,
        ngrams: list[torch.Tensor],
        words: list[torch.Tensor],
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        # b, s, h
        y: torch.Tensor = self.foot.forward(ngrams, words, word_spans)
        # b, h
        return y.mean(axis=1)

    def forward(
        self,
        ngrams: list[torch.Tensor],
        words: list[torch.Tensor],
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (torch.Tensor): ngrams, shape is batch_size, 1 or neighbor_size, ngram_len
            words (torch.Tensor): words, shape is (batch_size, 1 or neighbor_size, word_len)
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size*neighbor_size x (#subsequence)

        Returns:
            torch.Tensor: embedded vectors, shape is (batch_size, #subsequence, vocab_size)
        """
        y = self.get_sentence_vector(ngrams, words, word_spans)
        return self.head(y)
