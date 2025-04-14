import torch
from torch import nn


class SkipRNN(nn.Module):

    def __init__(
        self,
        hidden_dim_c: int,
        hidden_dim_skip: int,
        kernel: int,
        window_size: int,
        skip_window_size: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.hidden_dim_c = hidden_dim_c
        self.skip_window_size = skip_window_size
        self.p = (window_size - kernel + 1) // skip_window_size

        self.rnn = nn.GRU(hidden_dim_c, hidden_dim_skip)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        b, hc, _ = c.shape
        c = c[:, :, -int(self.p * self.skip_window_size) :]
        c = c.view(b, hc, self.p, self.skip_window_size)
        # percent, batch * skip window size, hc
        c = c.permute(2, 0, 3, 1).reshape(self.p, -1, self.hidden_dim_c)
        # D=1, batch * skip window size, hs
        _, y = self.rnn(c)
        y = self.dropout(y)
        # batch size, skip window size * hs
        y = y.view(b, -1)

        # B, hr + skip window size * hs
        return torch.cat([r, y], -1)


class TemporalAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # c: B, hc, ? # r: B, hr

        # B, 1, hr x B, hc, ? => B, 1, ?
        a = torch.bmm(r[:, None, :], c)
        # B, hc, ? x B, ?, 1 => B, hc
        y = torch.bmm(c, a.softmax(-1).transpose(-1, -2)).squeeze(-1)

        # B, hc
        return y


class Highway(nn.Module):

    def __init__(
        self,
        highway_window_size: int,
    ):
        super().__init__()
        self.highway_window_size = highway_window_size
        self.linear = nn.Linear(highway_window_size, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: B, 1, S, C # z: B, C

        output_size = x.size(-1)
        # B, highway window size, C
        x = x[:, 0, -self.highway_window_size :, :]
        # B * C, highway window size
        x = x.transpose(-1, -2).reshape(-1, self.highway_window_size)
        # B * C, 1
        y = self.linear(x)
        # B, C
        y = y.view(-1, output_size)

        # B, C
        return z + y
