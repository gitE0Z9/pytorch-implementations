import torch
from torch import nn


class GruClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        padding_idx: int | None = None,
    ):
        super(GruClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.embed(x)  # batch_size, seq_len, embed_dim
        # ot: batch_size, seq_len, bidirectional*hidden_dim
        # ht: bidirectional * layer_size, batch_size, hidden_dim
        _, (ht, _) = self.rnn(output)

        # the last layer's hidden state represents the paragraph
        output = self.fc(ht[-1])

        return output
