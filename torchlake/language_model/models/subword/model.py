from typing import Literal

import torch
from torch import nn

from torchlake.common.schemas.nlp import NLPContext

from ...constants.enum import LossType, NgramCombinationMethod, Word2VecModelType
from ..word2vec import Word2Vec
from .network import SubwordEmbedding


class SubwordLM(Word2Vec):

    def __init__(
        self,
        bucket_size: int,
        output_vocab_size: int,
        embed_dim: int,
        neighbor_size: int,
        model_type: Word2VecModelType,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        ngram_reduction: Literal["sum", "mean"] = "mean",
        combination: NgramCombinationMethod = NgramCombinationMethod.WORD_AND_NGRAM,
        context: NLPContext | None = None,
        input_vocab_size: int = 0,
    ):
        """Subword language model in paper [1607.04606] and extension in paper [1712.09405v1]

        Args:
            bucket_size (int): size of hash bucket
            output_vocab_size (int): size of vocabulary
            embed_dim (int): embedding dimension
            neighbor_size (int): how many tokens around the gram, i.e. context size - 1. Defaults to 1.
            model_type (Word2VecModelType): model type, either CBOW or SkipGram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CROSS_ENTROPY.
            ngram_reduction (Literal["sum", "mean"], optional): redution mode of ngrams. Defaults to "mean".
            combination (NgramCombinationMethod, optional): combination method of word vector and ngrams vectors. Defaults to NgramCombinationMethod.WORD_AND_NGRAM.
            context (NLPContext, optional): NLP context. Defaults to None.
            input_vocab_size (int, optional): size of a separate word embedding. Defaults to 0.
        """
        if context is None:
            context = NLPContext()

        self.bucket_size = bucket_size
        self.output_vocab_size = output_vocab_size
        self.embed_dim = embed_dim
        self.neighbor_size = neighbor_size
        self.model_type = model_type
        self.loss_type = loss_type
        self.ngram_reduction = ngram_reduction
        self.combination = combination
        self.context = context
        self.input_vocab_size = input_vocab_size
        super().__init__(input_vocab_size + bucket_size, output_vocab_size)

    # @property
    # def input_embedding(self) -> nn.Embedding:
    #     return self.foot

    # @property
    # def output_embedding(self) -> nn.Embedding:
    #     if self.loss_type == LossType.CROSS_ENTROPY:
    #         w = self.head.weight.detach()
    #         out_c, in_c = w.shape
    #         e = nn.Embedding(
    #             out_c,
    #             in_c,
    #             padding_idx=self.context.padding_idx,
    #         )
    #         e.weight.data.copy_(w)
    #         return e
    #     elif self.loss_type == LossType.NEGATIVE_SAMPLING:
    #         return self.head
    #     else:
    #         raise ValueError("only supports when loss type is ce or ns")

    def build_foot(self, _, **kwargs):
        self.foot: SubwordEmbedding = SubwordEmbedding(
            self.bucket_size,
            self.embed_dim,
            vocab_size=self.input_vocab_size,
            ngram_reduction=self.ngram_reduction,
            combination=self.combination,
            context=self.context,
        )

    # def build_blocks(self, **kwargs):
    #     if self.model_type == Word2VecModelType.CBOW:
    #         # aggregate context with "mean" as author(https://github.com/tmikolov/word2vec/blob/master/word2vec.c)
    #         # "sum" used in paper(https://arxiv.org/abs/1301.3781) is deprecated
    #         # b*subseq, 1, h
    #         self.blocks = lambda x, _: x.mean(1, keepdim=True)
    #     elif self.model_type == Word2VecModelType.SKIP_GRAM:
    #         # b*subseq, n, h
    #         self.blocks = lambda x, neighbor_size: x.repeat(1, neighbor_size, 1)
    #     else:
    #         raise ValueError("only supports when model type is cbow or skip gram")

    def build_blocks(self, **kwargs):
        super().build_blocks(**kwargs)
        if self.model_type == Word2VecModelType.CBOW:
            self.blocks = lambda x, _: x

    def build_neck(self, **kwargs):
        if self.model_type == Word2VecModelType.CBOW:
            # vlbl
            self.neck = nn.Conv2d(self.neighbor_size, 1, 1, bias=False)
        elif self.model_type == Word2VecModelType.SKIP_GRAM:
            # ivlbl
            self.neck = nn.Parameter(torch.ones((self.neighbor_size,)))
        else:
            raise ValueError("unrecognized model type")

    # def build_head(self, _, **kwargs):
    #     if self.loss_type == LossType.CROSS_ENTROPY:
    #         self.head = nn.Linear(
    #             self.embed_dim,
    #             self.vocab_size,
    #             # bias=False,
    #         )
    #     elif self.loss_type == LossType.NEGATIVE_SAMPLING:
    #         self.head = nn.Embedding(
    #             self.vocab_size,
    #             self.embed_dim,
    #             padding_idx=self.context.padding_idx,
    #         )
    #     elif self.loss_type == LossType.HIERARCHICAL_SOFTMAX:
    #         root = kwargs.pop("root")
    #         self.head = HierarchicalSoftmax(
    #             root,
    #             self.vocab_size,
    #             self.embed_dim,
    #             context=self.context,
    #         )
    #     else:
    #         raise ValueError

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
        word_spans: list[torch.Tensor],
        words: torch.Tensor | None = None,
        neighbor_size: int = 1,
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size*neighbor_size x (#grams)
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size*neighbor_size x (#subsequence)
            words (torch.Tensor, optional): word tokens, shape is batch_size, max_seq_len. Defaults to None.
            neighbor_size (int): how many tokens around the middle gram for skip gram model, i.e. context size - 1. Defaults to 1.

        Returns:
            torch.Tensor: output vector, shape is (batch_size*#subsequence, neighbor_size or 1, embedding_dim)
        """
        # b, 1 or neighbor_size, s, h
        y = self.get_word_vectors(ngrams, words, word_spans)
        y = self.blocks(y, neighbor_size)

        if self.model_type == Word2VecModelType.CBOW:
            # b*s, 1, h
            y = self.neck(y)
        elif self.model_type == Word2VecModelType.SKIP_GRAM:
            # b*s, neighbor_size, h
            y = y * self.neck.view(1, -1, 1)
        else:
            raise ValueError("only supports when model type is cbow or skip gram")

        # b*s, 1 or neighbor_size, o
        return self.head(y)
