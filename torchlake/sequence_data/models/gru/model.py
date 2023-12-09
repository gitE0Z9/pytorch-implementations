import torch
from torch import nn


class GruClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        padding_idx: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super(GruClassifier, self).__init__()
        self.factor = 2 if bidirectional else 1

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(
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

        # ot, (ht, ct)
        # ot: batch_size, seq_len, bidirectional*hidden_dim
        # ht: bidirectional * layer_size, batch_size, hidden_dim
        ot, (ht, _) = self.rnn(y)

        if self.fc.out_features <= 1:
            # the last layer's hidden state represents the paragraph
            y = self.fc(ht[-1])
        else:
            y = self.layer_norm(ot)
            y = self.fc(y)

        return y
