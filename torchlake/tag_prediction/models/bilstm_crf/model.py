import torch
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.sequence_data.models.lstm import LSTMDiscriminator

from .network import LinearCRF


class BiLSTM_CRF(ModelBase):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        context: NlpContext = NlpContext(),
    ) -> None:
        """BiLSTM-CRF [1508.01991]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int): dimension of hidden layer
            output_size (int, optional): output size. Defaults to 1.
            num_layers (int, optional): number of lstm layers. Defaults to 1.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.context = context
        super().__init__(vocab_size, output_size)

    def build_foot(self, vocab_size):
        self.foot = LSTMDiscriminator(
            vocab_size,
            self.embed_dim,
            self.hidden_dim,
            self.output_size,
            num_layers=self.num_layers,
            bidirectional=True,
            context=self.context,
            sequence_output=True,
        )

    def build_head(self, output_size):
        self.head = LinearCRF(output_size, self.context)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_score: bool = False,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input token tensor, shape is (batch_size, sequence_length)
            mask (torch.Tensor | None, optional): mask for padding index. Defaults to None.
            output_score (bool, optional): return score of viterbi path. Defaults to False.

        Returns:
            tuple[torch.Tensor] | torch.Tensor: lstm output when training, crf paths when inference
        """
        # B, S, O
        y = self.foot(x)

        if not self.training:
            y = self.head(y, mask, output_score)

        return y
