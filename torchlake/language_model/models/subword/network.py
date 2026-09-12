from typing import Literal

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean

from torchlake.common.schemas.nlp import NLPContext

from ...constants.enum import NgramCombinationMethod


class SubwordEmbedding(nn.Module):

    def __init__(
        self,
        bucket_size: int,
        embed_dim: int,
        vocab_size: int = 0,
        ngram_reduction: Literal["sum", "mean"] = "mean",
        combination: NgramCombinationMethod = NgramCombinationMethod.WORD_AND_NGRAM,
        context: NLPContext | None = None,
    ):
        """ngram embedding

        Args:
            bucket_size (int): size of hash bucket
            embed_dim (int): embedding dimension
            vocab_size (int, optional): size of separate word embedding. Defaults to 0.
            ngram_reduction (Literal["sum", "mean"], optional): redution mode of ngrams. Defaults to "mean".
            combination (NgramCombinationMethod, optional): combination method of word vector and ngrams vectors. Defaults to NgramCombinationMethod.WORD_AND_NGRAM.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        super().__init__()
        self.bucket_size = bucket_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.combination = combination
        self.context = context
        self.set_reduction(ngram_reduction)

        self.embeddings = nn.Embedding(bucket_size, embed_dim)
        self.word_embeddings = nn.Embedding(
            vocab_size if vocab_size > 0 else len(context.special_tokens),
            embed_dim,
            padding_idx=context.padding_idx,
        )

    def set_reduction(self, reduction: Literal["sum", "mean"] = "mean"):
        """set reduction method

        Args:
            reduction (Literal[mean] | Literal[sum], optional): redution mode. Defaults to "mean".

        Raises:
            ValueError: reduction must be either 'sum' or 'mean'
        """
        if reduction == "sum":
            self.reduction = scatter_add
        elif reduction == "mean":
            self.reduction = scatter_mean
        else:
            raise ValueError("reduction must be either 'sum' or 'mean'")

    def expand_word_indices(self, word_spans: list[torch.Tensor]) -> torch.Tensor:
        """expand word spans to word indices

        Args:
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size x #words

        Returns:
            torch.Tensor: expanded word indices
        """
        word_spans: torch.Tensor = torch.cat(word_spans, -1)
        # sum of word_spans, 1
        return (
            torch.arange(len(word_spans), device=word_spans.device)
            .repeat_interleave(word_spans)
            .unsqueeze_(-1)
        )

    def forward(
        self,
        ngrams: list[torch.Tensor],
        word_spans: list[torch.Tensor],
        words: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size x #grams
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size x #words
            words (torch.Tensor, optional): word tokens, shape is batch_size, max_seq_len. Defaults to None.

        Returns:
            torch.Tensor: embedded vectors, shape is (batch_size, max_seq_len, embedding dimension)
        """
        # sum of ngram_seq_len
        ngrams = torch.cat(ngrams, -1)
        # sum of ngram_seq_len
        _word_spans = self.expand_word_indices(word_spans)
        # sum of ngram_seq_len, embed_dim
        y = self.embeddings(ngrams)
        # sum of seq_len, embed_dim
        y = self.reduction(y, _word_spans, 0)

        seq_lens = [word_span.size(-1) for word_span in word_spans]

        pad_vector: torch.Tensor = self.word_embeddings(
            torch.tensor([self.context.padding_idx], device=y.device, dtype=torch.long)
        )
        y = [
            # batch size x max_seq_len, embed_dim
            torch.cat(
                [
                    ele,
                    pad_vector.expand(
                        self.context.max_seq_len - ele.size(-2), ele.size(-1)
                    ),
                ],
                -2,
            )
            # batch size x seq_len, embed_dim
            for ele in y.split(seq_lens)
        ]

        # batch size, max_seq_len, embed_dim
        y = torch.stack(y)

        if self.combination == NgramCombinationMethod.WORD_AND_NGRAM:
            assert (
                words is not None
            ), "when combination method is word and ngrams, words must be provided"
            # batch_size, max_seq_len, embed_dim
            y += (
                self.word_embeddings(words)  # dict word
                if self.vocab_size > 0
                else self.embeddings(words)  # hashed word
            )

        return y
