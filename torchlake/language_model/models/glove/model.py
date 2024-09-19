import torch
from torch import nn

from torchlake.common.schemas.nlp import NlpContext


class GloVe(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context: NlpContext = NlpContext(),
    ):
        """GloVe: Global Vectors for Word Representation

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        super(GloVe, self).__init__()
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
        self.word_bias = nn.Parameter(torch.zeros((vocab_size,)))
        self.context_bias = nn.Parameter(torch.zeros((vocab_size,)))

    def forward(self, gram: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            gram (torch.Tensor): shape is batch_size*subseq_len, 1
            context (torch.Tensor): shape is batch_size*subseq_len, neighbor_size

        Returns:
            torch.Tensor: w_i * w_j + b_i + b_j
        """
        # b*seq, 1, h # b*seq, neighbor_size, h
        w_e, c_e = self.word_embed(gram), self.context_embed(context)
        # b*seq, neighbor_size + b*seq, 1 + b*seq, neighbor_size => b*seq, neighbor_size
        return (
            torch.einsum("sxh,syh->sy", w_e, c_e)
            + self.word_bias[gram]
            + self.context_bias[context]
        )
