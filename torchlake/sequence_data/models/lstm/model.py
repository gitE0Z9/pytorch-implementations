import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchlake.common.schemas.nlp import NlpContext


class LstmClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
    ):
        super(LstmClassifier, self).__init__()
        self.factor = 2 if bidirectional else 1
        self.context = context

        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * self.factor)
        self.fc = nn.Linear(hidden_dim * self.factor, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, seq_len, embed_dim
        y = self.embed(x)

        if self.embed.padding_idx is not None:
            y = pack_padded_sequence(
                y,
                x.ne(self.embed.padding_idx).sum(dim=1).long().detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

        # ot, (ht, ct)
        # ot: batch_size, seq_len, bidirectional*hidden_dim
        # ht: bidirectional * layer_size, batch_size, hidden_dim
        ot, (ht, _) = self.rnn(y)

        if isinstance(ot, PackedSequence):
            ot, _ = pad_packed_sequence(
                ot,
                batch_first=True,
                total_length=self.context.max_seq_len,
            )

        if self.fc.out_features <= 1:
            # the last layer's hidden state represents the paragraph
            y = self.fc(ht[-1])
        else:
            y = self.layer_norm(ot)
            y = self.fc(y)

        return y
