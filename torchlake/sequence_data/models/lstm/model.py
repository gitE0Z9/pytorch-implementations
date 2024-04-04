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
        is_token: bool = False,
    ):
        super(LstmClassifier, self).__init__()
        self.factor = 2 if bidirectional else 1
        self.context = context
        self.is_token = is_token

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

    def feature_extract(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        state = (h, c) if h is not None or c is not None else None

        return self.rnn(y, state)

    def classify(
        self,
        ot: torch.Tensor,
        ht: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_token:
            y = self.layer_norm(ot)
            y = self.fc(y)
        else:
            # the deepest layer's hidden state recept the whole sequence
            y = torch.cat([ht[-2], ht[-1]], -1) if self.factor == 2 else ht[-1]
            y = self.fc(y)

        return y

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ot, (ht, _) = self.feature_extract(x, h, c)

        if isinstance(ot, PackedSequence):
            ot, _ = pad_packed_sequence(
                ot,
                batch_first=True,
                total_length=self.context.max_seq_len,
            )

        return self.classify(ot, ht)
