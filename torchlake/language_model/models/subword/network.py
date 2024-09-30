from typing import Literal

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torchlake.common.schemas.nlp import NlpContext

from ...constants.enum import NgramCombinationMethod


class SubwordEmbedding(nn.Module):

    def __init__(
        self,
        bucket_size: int,
        embed_dim: int,
        ngram_reduction: Literal["sum", "mean"] = "mean",
        combination: NgramCombinationMethod = NgramCombinationMethod.WORD_AND_NGRAM,
        context: NlpContext = NlpContext(),
    ):
        """ngram embedding

        Args:
            bucket_size (int): size of hash bucket
            embed_dim (int): embedding dimension
            ngram_reduction (Literal["sum", "mean"], optional): redution mode of ngrams. Defaults to "mean".
            combination (NgramCombinationMethod, optional): combination method of word vector and ngrams vectors. Defaults to NgramCombinationMethod.WORD_AND_NGRAM.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(SubwordEmbedding, self).__init__()
        self.embeddings = nn.Embedding(bucket_size, embed_dim)
        self.special_tokens_embedding = nn.Embedding(
            len(context.special_tokens),
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.combination = combination
        self.context = context

        self.set_reduction(ngram_reduction)

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

    def forward(
        self,
        ngrams: list[torch.Tensor],
        words: torch.Tensor,
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size x #grams
            words (torch.Tensor): word tokens, shape is batch_size, max_seq_len
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size x #words

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
        y = self.reduction(y, _word_spans, -2)

        seq_lens = [word_span.size(-1) for word_span in word_spans]

        pad_vector: torch.Tensor = self.special_tokens_embedding(
            torch.LongTensor([self.context.padding_idx]).to(y.device)
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
            # batch_size, max_seq_len, embed_dim
            y += self.embeddings(words)

        return y

    def expand_word_indices(self, word_spans: list[torch.Tensor]) -> torch.Tensor:
        """expand word spans to word indices

        Args:
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size x #words

        Returns:
            torch.Tensor: expanded word indices
        """
        word_spans: torch.Tensor = torch.cat(word_spans, -1)
        return (
            torch.arange(len(word_spans))
            .to(word_spans.device)
            .repeat_interleave(word_spans)
            .unsqueeze_(-1)
        )
        # _word_spans = []
        # _prev_word_spans = 0
        # for word_span in word_spans:
        #     l = word_span.size(-1)
        #     _word_span = (
        #         torch.arange(l).to(word_span.device).repeat_interleave(word_span)
        #         + _prev_word_spans
        #     )
        #     _prev_word_spans += l
        #     _word_spans.append(_word_span)

        # return torch.cat(_word_spans, -1).unsqueeze_(-1)
