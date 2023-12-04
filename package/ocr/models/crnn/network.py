import torch
from torch import nn


class RecurrnetBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layer: int = 1, bidirect:bool=False):
        super(RecurrnetBlock, self).__init__()

        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=num_layer, bidirectional=bidirect)
        self.linear = nn.Linear(hidden_dim * (2 if bidirect else 1), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.rnn(x)
        t, b, h = o.shape
        o = o.view(-1, h)
        o = self.linear(o)
        o = o.view(t, b, -1)

        return o
