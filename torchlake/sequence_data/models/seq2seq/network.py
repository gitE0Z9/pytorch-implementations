import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

from ..lstm.model import LstmClassifier


class Seq2SeqEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
        is_token: bool = True,
    ):
        """
        將欲翻譯句子轉為隱向量
        """
        super(Seq2SeqEncoder, self).__init__()
        self.rnn = LstmClassifier(
            vocab_size,
            embed_dim,
            hidden_dim,
            output_size,
            num_layers,
            bidirectional,
            context,
            is_token=is_token,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        _, state = self.rnn.feature_extract(x)
        return state


class Seq2SeqDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
        is_token: bool = True,
    ):
        """
        將隱向量與目標句子轉為欲翻譯句子
        """
        super(Seq2SeqDecoder, self).__init__()
        self.rnn = LstmClassifier(
            vocab_size,
            embed_dim,
            hidden_dim,
            output_size,
            num_layers,
            bidirectional,
            context,
            is_token=is_token,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return self.rnn(x, h, c)
