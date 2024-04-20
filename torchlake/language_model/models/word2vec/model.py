import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.language_model.constants.enum import LossType, ModelType


class Cbow(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context: NlpContext = NlpContext(),
    ):
        """Continuous bag of words model, use context to predict gram

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(Cbow, self).__init__()

        self.embeddings = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)

        Returns:
            torch.Tensor: embedded vectors, shape is (batch_size, 1, #subsequence, vocab_size)
        """
        # B, 1, subseq, embed_dim
        return self.embeddings(x).sum(dim=1, keepdim=True)


class SkipGram(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_size: int,
        context: NlpContext = NlpContext(),
    ):
        """Skip gram model, use gram to predict context

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            context_size (int): size of each subsequence
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(SkipGram, self).__init__()
        self.context_size = context_size

        self.embeddings = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)

        Returns:
            torch.Tensor: embedded vectors, shape is (batch size, neighbor_size, #subsequence, embedding dimension)
        """
        # B, neighbor, subseq, embed_dim
        return self.embeddings(x).repeat(1, self.context_size - 1, 1, 1)


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_size: int,
        model_type: ModelType,
        loss_type: LossType = LossType.CE,
        context: NlpContext = NlpContext(),
    ):
        """Word2Vec model

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            context_size (int): size of each subsequence
            model_type (ModelType): model type, either Cbow or Skipgram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CE.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(Word2Vec, self).__init__()
        self.loss_type = loss_type
        self.fc = nn.Identity()
        self.model = self._build_model(
            model_type,
            loss_type,
            vocab_size,
            embed_dim,
            context_size,
            context,
        )

    def _build_model(
        self,
        model_type: ModelType,
        loss_type: LossType,
        vocab_size: int,
        embed_dim: int,
        context_size: int,
        context: NlpContext = NlpContext(),
    ) -> Cbow | SkipGram:
        """build model with options

        Args:
            model_type (ModelType): model type, either Cbow or Skipgram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CE.
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            context_size (int): size of each subsequence
            context (NlpContext, optional): context object. Defaults to NlpContext().

        Returns:
            Cbow | SkipGram: nn.Module
        """

        if model_type == ModelType.CBOW:
            model = Cbow(vocab_size, embed_dim, context)
        elif model_type == ModelType.SKIP_GRAM:
            model = SkipGram(vocab_size, embed_dim, context_size, context)

        if loss_type == LossType.CE:
            self.fc = nn.Linear(embed_dim, vocab_size)

        return model

    @property
    def embeddings(self) -> nn.Module:
        return self.model.embeddings

    @staticmethod
    def subsampling(
        x: torch.Tensor,
        subsampling_probs: torch.Tensor,
    ) -> torch.Tensor:
        """1310.4546 p.4
        subsampling tokens with the probability of word frequency formula

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)
            subsampling_probs (torch.Tensor): probability distribution of each token with formula in paper

        Returns:
            torch.Tensor: subsampled one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)
        """
        return x * subsampling_probs[x].bernoulli()

    def forward(
        self,
        x: torch.Tensor,
        word_probs: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)
            word_probs (dict[int, float] | None, optional): probability distribution of each token with formula in paper. Defaults to None.

        Returns:
            torch.Tensor: output vector, shape is (batch_size, neighbor_size, #subsequence, embedding_dim) or (batch_size, neighbor_size, #subsequence, vocab_size)
        """
        if word_probs is not None:
            x = self.subsampling(x, word_probs).long()

        y = self.model(x)

        # B, neighbor, subseq, embed_dim | B, neighbor, subseq, vocab_size
        y = self.fc(y)

        if self.loss_type == LossType.CE:
            y = y.permute(0, -1, 1, 2)

        return y
