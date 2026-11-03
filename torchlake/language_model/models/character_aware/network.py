import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from torchlake.common.models.multikernel_conv import MultiKernelConvModule
from torchlake.common.schemas.nlp import NLPContext


class CharacterCNNEmbedding(nn.Module):
    def __init__(
        self,
        char_size: int,
        embed_dim: int,
        hidden_dims: list[int],
        kernels: list[int],
        max_char_size: int = 65,
        context: NLPContext | None = None,
    ):
        if context is None:
            context = NLPContext()

        super().__init__()
        self.max_char_size = max_char_size
        self.context = context

        self.embedding = nn.Embedding(
            char_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.layers = MultiKernelConvModule(
            embed_dim,
            hidden_dims,
            kernels,
            activation=nn.Tanh(),
            reduction="max",
            dimension="1d",
            concat_output=True,
        )

    def forward(
        self,
        ngrams: list[torch.Tensor],
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, length is batch size, the shape of each tensor is #ngram in a sequence
            word_spans (list[torch.Tensor]): word lengths, length is batch size, the shape of each tensor is seq_len

        Returns:
            torch.Tensor: shape is how many words in the batch, sum of hidden dims
        """
        # all words in the batch x (char_size,)
        word_stack = torch.cat(ngrams).split(torch.cat(word_spans).tolist())
        # all words in the batch x (at most max_char_size,)
        word_stack = [w[: self.max_char_size] for w in word_stack]
        # all words in the batch, at most max_char_size
        y = pad_sequence(
            word_stack,
            batch_first=True,
            padding_value=self.context.padding_idx,
        )

        # all words in the batch, at most max_char_size, embed_dim
        y = self.embedding(y)
        # all words in the batch, sum of hidden dims
        y = self.layers(y.transpose(1, 2))

        return y
