from typing import Literal

import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

from ...constants.enum import LossType, NgramCombinationMethod, Word2VecModelType
from .network import SubwordEmbedding


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
