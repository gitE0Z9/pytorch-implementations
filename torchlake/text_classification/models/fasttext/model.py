from typing import Literal

import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import HuffmanNode
from torchlake.language_model.constants.enum import LossType, NgramCombinationMethod
from torchlake.language_model.models.subword import SubwordEmbedding
from torchlake.language_model.models.word2vec import HierarchicalSoftmax


class FastText(ModelBase):
    def __init__(
        self,
        bucket_size: int,
        embed_dim: int,
        output_size: int,
        input_vocab_size: int = 0,
        ngram_reduction: Literal["sum", "mean"] = "mean",
        combination: NgramCombinationMethod = NgramCombinationMethod.WORD_AND_NGRAM,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        context: NLPContext | None = None,
        root: HuffmanNode | None = None,
    ):
        """FastText [1607.04606]

        Args:
            bucket_size (int): size of hash bucket
            embed_dim (int): embedding dimension
            output_size (int, optional): number of features of output. Defaults to 1.
            input_vocab_size (int, optional): size of a separate word embedding. Defaults to 0.
            ngram_reduction (Literal["sum", "mean"], optional): redution mode of ngrams. Defaults to "mean".
            combination (NgramCombinationMethod, optional): combination method of word vector and ngrams vectors. Defaults to NgramCombinationMethod.WORD_AND_NGRAM.
            loss_type (LossType, optional): loss type, cross entropy, hierarchical softmax. Defaults to LossType.CROSS_ENTROPY.
            context (NLPContext, optional): NLP context. Defaults to None.
            root: HuffmanNode | None = None,
        """
        if context is None:
            context = NLPContext()

        self.bucket_size = bucket_size
        self.embed_dim = embed_dim
        self.input_vocab_size = input_vocab_size
        self.ngram_reduction = ngram_reduction
        self.combination = combination
        self.loss_type = loss_type
        self.context = context
        super().__init__(
            bucket_size,
            output_size,
            head_kwargs={
                "root": root,
            },
        )

    @property
    def embeddings(self) -> nn.Embedding:
        return self.foot.embeddings

    def build_foot(self, bucket_size: int):
        self.foot = SubwordEmbedding(
            bucket_size,
            self.embed_dim,
            vocab_size=self.input_vocab_size,
            ngram_reduction=self.ngram_reduction,
            combination=self.combination,
            context=self.context,
        )

    def build_head(self, output_size: int, **kwargs):
        if self.loss_type == LossType.CROSS_ENTROPY:
            self.head = nn.Linear(self.embed_dim, output_size)
        elif self.loss_type == LossType.HIERARCHICAL_SOFTMAX:
            root = kwargs.pop("root")
            self.head = HierarchicalSoftmax(
                root,
                output_size,
                self.embed_dim,
                context=self.context,
            )
        else:
            raise ValueError

    def get_sentence_vector(
        self,
        ngrams: list[torch.Tensor],
        words: list[torch.Tensor],
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        # b, s, h
        y: torch.Tensor = self.foot(ngrams, word_spans, words)
        # b, h
        return y.mean(axis=1)

    def forward(
        self,
        ngrams: list[torch.Tensor],
        words: torch.Tensor,
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngrams, shape is batch_size x (#ngram)
            words (torch.Tensor): words, shape is batch_size, seq_len
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size x (#word)

        Returns:
            torch.Tensor: shape is (batch_size, output_size)
        """
        y = self.get_sentence_vector(ngrams, words, word_spans)

        if self.loss_type == LossType.HIERARCHICAL_SOFTMAX and self.training:
            return y

        return self.head(y)
