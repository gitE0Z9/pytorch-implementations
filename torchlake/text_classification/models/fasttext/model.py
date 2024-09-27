import torch
import torch_scatter
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.language_model.constants.enum import LossType


class FastText(nn.Module):
    def __init__(
        self,
        bucket_size: int,
        embed_dim: int,
        output_size: int,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        context: NlpContext = NlpContext(),
    ):
        """Fasttext [1607.04606]

        Args:
            bucket_size (int): size of hash bucket
            embed_dim (int): embedding dimension
            output_size (int, optional): number of features of output. Defaults to 1.
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CE.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(FastText, self).__init__()

        self.embeddings = nn.Embedding(
            bucket_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )

        if loss_type == LossType.CE:
            self.fc = nn.Linear(embed_dim, output_size)
        else:
            self.fc = nn.Identity()

    def get_words_vector(
        self,
        x: torch.Tensor,
        word_incices: torch.Tensor,
    ) -> torch.Tensor:
        y = self.embeddings(x)
        return torch_scatter.scatter_add(y, word_incices, 1)

    def get_sentence_vector(
        self,
        x: torch.Tensor,
        word_incices: torch.Tensor,
    ) -> torch.Tensor:
        y = self.get_words_vector(x, word_incices)
        return y.mean(axis=1)

    def forward(self, x: torch.Tensor, word_incices: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size, neighbor_size, #subsequence)

        Returns:
            torch.Tensor: embedded vectors, shape is (batch_size, #subsequence, vocab_size)
        """
        y = self.get_sentence_vector(x, word_incices)
        return self.fc(y)
