import torch
from torch import nn

from torchlake.common.schemas.nlp import NlpContext


class VLBL(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        neighbor_size: int,
        context: NlpContext = NlpContext(),
    ):
        """vector log bilinear model in paper [Learning word embeddings efficiently with noise-contrastive estimation]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            neighbor_size (int): how many tokens around the gram, i.e. context size - 1. Defaults to 1.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        super(VLBL, self).__init__()
        self.word_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.context_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        # context_position_weights can also be average
        # here we use conv to implement so no need to manager parameter
        self.position_weights = nn.Conv2d(neighbor_size, 1, 1)

        self.bias = nn.Parameter(torch.zeros((vocab_size,)))

    def forward(self, context: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
        """vLBL forward

        Args:
            gram (torch.Tensor): center word
            context (torch.Tensor): surrounding words

        Returns:
            torch.Tensor: similarity from context to gram
        """
        # b, 1, s, h / b, c-1, s, h
        w_e, c_e = self.word_embed(gram), self.context_embed(context)
        # (b, 1, s, h x b, 1, s, h) + b, 1, s => b, 1, s
        return (
            torch.einsum("bxsh,bxsh->bxs", w_e, self.position_weights(c_e))
            + self.bias[gram]
        )


class IVLBL(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        neighbor_size: int,
        context: NlpContext = NlpContext(),
    ):
        """inverse vector log bilinear model in paper [Learning word embeddings efficiently with noise-contrastive estimation]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            neighbor_size (int): how many tokens around the gram, i.e. context size - 1. Defaults to 1.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        super(IVLBL, self).__init__()
        self.word_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.context_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.position_weights = nn.Parameter(torch.zeros((neighbor_size,)))

        self.bias = nn.Parameter(torch.zeros((vocab_size,)))

    def forward(self, gram: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """ivLBL forward

        Args:
            gram (torch.Tensor): center word
            context (torch.Tensor): surrounding words

        Returns:
            torch.Tensor: similarity from gram to context
        """
        # b, 1, s, h / b, c-1, s, h
        w_e, c_e = self.word_embed(gram), self.context_embed(context)
        # b, 1, s, h * 1, c-1, 1, 1 => b, c-1, s,h
        w_e = w_e * self.position_weights.view(1, -1, 1, 1)
        # (b, c-1, s, h x b, c-1, s, h) + b, c-1, s => b, c-1, s
        return torch.einsum("bxsh,bxsh->bxs", w_e, c_e) + self.bias[context]
