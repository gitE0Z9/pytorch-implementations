import torch
from torch import nn
from torchlake.sequence_data.models.base.wrapper import (
    SequenceModelFullFeatureExtractor,
)
from torchlake.common.schemas.nlp import NlpContext


class Rcnn(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        context: NlpContext = NlpContext(),
    ):
        super(Rcnn, self).__init__()
        self.context = context
        self.rnn = SequenceModelFullFeatureExtractor(
            vocab_size,
            embed_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            context=context,
            model_class=nn.RNN,
        )
        self.conv = nn.Conv1d(2 * hidden_dim + embed_dim, hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, s, e
        embedded = self.rnn.embed(x)
        # b, s, 2*h
        # XXX: computed twice
        context, _ = self.rnn(x)
        left_context, right_context = context.chunk(2, -1)
        # b, s, 2*h + e
        y = torch.cat([left_context, embedded, right_context], -1)
        # b, 2*h + e, s
        y = y.transpose(-1, -2)
        # b, h, s
        y = self.conv(y).tanh()
        # b, h
        y = y.max(-1)[0]

        return self.fc(y)
