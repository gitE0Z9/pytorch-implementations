from typing import Literal

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torchlake.common.schemas.nlp import NlpContext

from ...constants.enum import LossType, Word2VecModelType
from .enum import NgramCombinationMethod


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
        _word_spans = []
        _prev_word_spans = 0
        for word_span in word_spans:
            l = word_span.size(-1)
            _word_span = (
                torch.arange(l).to(word_span.device).repeat_interleave(word_span)
                + _prev_word_spans
            )
            _prev_word_spans += l
            _word_spans.append(_word_span)

        return torch.cat(_word_spans, -1).unsqueeze_(-1)


class SubwordLM(nn.Module):

    def __init__(
        self,
        bucket_size: int,
        vocab_size: int,
        embed_dim: int,
        model_type: Word2VecModelType,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        ngram_reduction: Literal["sum", "mean"] = "mean",
        combination: NgramCombinationMethod = NgramCombinationMethod.WORD_AND_NGRAM,
        context: NlpContext = NlpContext(),
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
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(SubwordLM, self).__init__()
        self.model_type = model_type

        self.embeddings = SubwordEmbedding(
            bucket_size,
            embed_dim,
            ngram_reduction=ngram_reduction,
            combination=combination,
            context=context,
        )

        self.fc = (
            nn.Linear(embed_dim, vocab_size)
            if loss_type == LossType.CROSS_ENTROPY
            else nn.Identity()
        )

        self._set_head()

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
        # n, 1 or neighbor_size, s, h
        y = self.get_embedding_vector(ngrams, words, word_spans, batch_size)
        return self.head(y, target_neighbor_size)

        # if self.model_type == Word2VecModelType.CBOW:
        #     y = y.mean(1, keepdim=True)
        #     return self.fc(y)
        # elif self.model_type == Word2VecModelType.SKIP_GRAM:
        #     y = self.fc(y)
        #     return y.repeat(1, neighbor_size, 1, 1)

    def get_embedding_vector(
        self,
        ngrams: list[torch.Tensor],
        words: torch.Tensor,
        word_spans: list[torch.Tensor],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """get embedding vector of ngrams

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size*neighbor_size x (#grams)
            words (torch.Tensor): word tokens, shape is batch_size, neighbor_size #subsequence)
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size*neighbor_size x (#subsequence)
            batch_size (int, optional): size of batch. Defaults to 1.

        Returns:
            torch.Tensor: embedding vector of ngrams
        """
        # batch_size * 1 or neighbor_size, s, h
        y: torch.Tensor = self.embeddings(ngrams, words, word_spans)
        n, seq_len, embed_dim = y.shape
        # batch_size, 1 or neighbor_size, s, h
        return y.view(batch_size, n // batch_size, seq_len, embed_dim)

    def _set_head(self):
        self.head = {
            Word2VecModelType.CBOW: self._forward_cbow,
            Word2VecModelType.SKIP_GRAM: self._forward_sg,
        }[self.model_type]

    def _forward_cbow(self, y: torch.Tensor, _: int = 1) -> torch.Tensor:
        y = y.mean(1, keepdim=True)
        return self.fc(y)

    def _forward_sg(self, y: torch.Tensor, neighbor_size: int = 1) -> torch.Tensor:
        y = self.fc(y)
        return y.repeat(1, neighbor_size, 1, 1)
