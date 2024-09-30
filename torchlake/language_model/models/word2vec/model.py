from typing import Literal
import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.language_model.constants.enum import LossType, Word2VecModelType


class CBOW(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int | None = None,
        reduction: Literal["sum", "mean"] = "mean",
    ):
        """Continuous bag of words model, use context to predict gram

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            padding_idx (int | None, optional): index of padding token. Defaults to None.
            reduction (Literal[mean] | Literal[sum], optional): redution mode. Defaults to "mean".
        """
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.set_reduction(reduction)

    def set_reduction(self, reduction: Literal["sum", "mean"] = "mean"):
        """set reduction method

        Args:
            reduction (Literal[mean] | Literal[sum], optional): redution mode. Defaults to "mean".

        Raises:
            ValueError: reduction must be either 'sum' or 'mean'
        """
        if reduction == "sum":
            self.reduction = lambda x: x.sum(dim=1, keepdim=True)
        elif reduction == "mean":
            self.reduction = lambda x: x.mean(dim=1, keepdim=True)
        else:
            raise ValueError("reduction must be either 'sum' or 'mean'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        aggregate context with "mean" as author(https://github.com/tmikolov/word2vec/blob/master/word2vec.c)

        "sum" used in paper(https://arxiv.org/abs/1301.3781) might be deprecated

        Args:
            x (torch.Tensor): context tokens, shape is (batch_size, neighbor_size, #subsequence)

        Returns:
            torch.Tensor: embedded vectors, shape is (batch_size, 1, #subsequence, vocab_size)
        """
        # B, neighbor_size, subseq, embed_dim
        y = self.embeddings(x)
        # B, 1, subseq, embed_dim
        return self.reduction(y)


class SkipGram(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int | None = None,
    ):
        """Skip gram model, use gram to predict context

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            padding_idx (int | None, optional): index of padding token. Defaults to None.
        """
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): center tokens, shape is (batch_size, 1, #subsequence)

        Returns:
            torch.Tensor: embedded vectors, shape is (batch size, 1, #subsequence, embedding dimension)
        """
        # B, 1, subseq, embed_dim
        # repeat neighbor_size in wrapper
        return self.embeddings(x)


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        model_type: Word2VecModelType,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        context: NlpContext = NlpContext(),
    ):
        """Word2Vec model

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            model_type (Word2VecModelType): model type, either CBOW or SkipGram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CROSS_ENTROPY.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(Word2Vec, self).__init__()
        self.context = context
        self.model_type = model_type
        self.loss_type = loss_type
        self.fc = nn.Identity()
        self.model = self._build_model(
            model_type,
            loss_type,
            vocab_size,
            embed_dim,
            context,
        )

    def _build_model(
        self,
        model_type: Word2VecModelType,
        loss_type: LossType,
        vocab_size: int,
        embed_dim: int,
        context: NlpContext = NlpContext(),
    ) -> CBOW | SkipGram:
        """build model with options

        Args:
            model_type (Word2VecModelType): model type, either CBOW or SkipGram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax.
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            context (NlpContext, optional): context object. Defaults to NlpContext().

        Returns:
            Cbow | SkipGram: nn.Module
        """
        model_mapping = {
            Word2VecModelType.CBOW: CBOW,
            Word2VecModelType.SKIP_GRAM: SkipGram,
        }
        model_cls = model_mapping[model_type]

        model = model_cls(vocab_size, embed_dim, context.padding_idx)

        if loss_type == LossType.CROSS_ENTROPY:
            self.fc = nn.Linear(embed_dim, vocab_size)

        return model

    @property
    def embeddings(self) -> nn.Embedding:
        return self.model.embeddings

    @staticmethod
    def subsampling(
        x: torch.Tensor,
        subsampling_probs: torch.Tensor,
        unk_idx: int,
    ) -> torch.Tensor:
        """1310.4546 p.4
        subsampling tokens with the probability of word frequency formula

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)
            subsampling_probs (torch.Tensor): probability distribution of each token with formula in paper
            unk_idx (int): unknown index as masked index

        Returns:
            torch.Tensor: subsampled one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)
        """
        return x.masked_fill_(~subsampling_probs[x].bernoulli().bool(), unk_idx)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_size: int = 1,
        word_probs: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)
            neighbor_size (int): how many tokens around the middle gram for skip gram model, i.e. context size - 1. Defaults to 1.
            word_probs (dict[int, float] | None, optional): probability distribution of each token with formula in paper. Defaults to None.

        Returns:
            torch.Tensor: output vector, shape is (batch_size, neighbor_size, #subsequence, embedding_dim) or (batch_size, neighbor_size, #subsequence, vocab_size)
        """
        if word_probs is not None:
            x = self.subsampling(x, word_probs, self.context.unk_idx).long()

        y: torch.Tensor = self.model(x)

        # To keep forward interface the same
        # move repeating here
        if self.model_type == Word2VecModelType.SKIP_GRAM:
            y = y.repeat(1, neighbor_size, 1, 1)

        # lossType != CROSS_ENTROPY => B, neighbor, subseq, embed_dim
        # lossType == CROSS_ENTROPY => B, neighbor, subseq, vocab_size
        return self.fc(y)
