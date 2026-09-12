from typing import Literal

import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext

from ...constants.enum import LossType, NgramCombinationMethod, Word2VecModelType
from .network import SubwordEmbedding


class SubwordLM(ModelBase):

    def __init__(
        self,
        bucket_size: int,
        vocab_size: int,
        embed_dim: int,
        model_type: Word2VecModelType,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        ngram_reduction: Literal["sum", "mean"] = "mean",
        combination: NgramCombinationMethod = NgramCombinationMethod.WORD_AND_NGRAM,
        context: NLPContext | None = None,
    ):
        """Subword language model in paper [1607.04606] and extension in paper [1712.09405v1]

        Args:
            bucket_size (int): size of hash bucket
            vocab_size (int): size of vocabulary
            embed_dim (int): embedding dimension
            model_type (Word2VecModelType): model type, either CBOW or SkipGram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CROSS_ENTROPY.
            ngram_reduction (Literal["sum", "mean"], optional): redution mode of ngrams. Defaults to "mean".
            combination (NgramCombinationMethod, optional): combination method of word vector and ngrams vectors. Defaults to NgramCombinationMethod.WORD_AND_NGRAM.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.bucket_size = bucket_size
        self.embed_dim = embed_dim
        self.model_type = model_type
        self.loss_type = loss_type
        self.ngram_reduction = ngram_reduction
        self.combination = combination
        self.context = context
        super().__init__(1, vocab_size)

    @property
    def embeddings(self) -> nn.Embedding:
        return self.foot.embeddings

    def build_foot(self, _, **kwargs):
        self.foot: SubwordEmbedding = SubwordEmbedding(
            self.bucket_size,
            self.embed_dim,
            ngram_reduction=self.ngram_reduction,
            combination=self.combination,
            context=self.context,
        )

    def build_head(self, output_size, **kwargs):
        self.head = (
            nn.Linear(self.embed_dim, output_size)
            if self.loss_type == LossType.CROSS_ENTROPY
            else nn.Identity()
        )

    def get_word_vectors(
        self,
        ngrams: list[torch.Tensor],
        words: torch.Tensor,
        word_spans: list[torch.Tensor],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """get embedded vector of words

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size*neighbor_size x (#grams)
            words (torch.Tensor): word tokens, shape is batch_size, neighbor_size #subsequence)
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size*neighbor_size x (#subsequence)
            batch_size (int, optional): size of batch. Defaults to 1.

        Returns:
            torch.Tensor: embedded vectors of words
        """
        # batch_size * 1 or neighbor_size, s, h
        y: torch.Tensor = self.foot(ngrams, word_spans, words)
        n, seq_len, embed_dim = y.shape
        # batch_size, 1 or neighbor_size, s, h
        return y.view(batch_size, n // batch_size, seq_len, embed_dim)

    def forward(
        self,
        ngrams: list[torch.Tensor],
        words: torch.Tensor,
        word_spans: list[torch.Tensor],
        batch_size: int = 1,
        target_neighbor_size: int = 1,
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size*neighbor_size x (#grams)
            words (torch.Tensor): word tokens, shape is batch_size, neighbor_size #subsequence)
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size*neighbor_size x (#subsequence)
            batch_size (int, optional): size of batch. Defaults to 1.
            target_neighbor_size (int, optional): size of neighbor in the window. Defaults to 1.

        Returns:
            torch.Tensor: embedding vectors of contexts for CBOW or of gram for SkipGram
        """
        # b, 1 or neighbor_size, s, h
        y = self.get_word_vectors(ngrams, words, word_spans, batch_size)
        if self.model_type == Word2VecModelType.CBOW:
            y = y.mean(1, keepdim=True)

        # b, 1, s, o
        y = self.head(y)

        if self.model_type == Word2VecModelType.SKIP_GRAM:
            # b, neighbor_size, s, o
            y = y.repeat(1, target_neighbor_size, 1, 1)

        # b, 1 or neighbor_size, s, o
        return y
