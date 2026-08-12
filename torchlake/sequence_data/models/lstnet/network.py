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
        """Skip RNN

        Args:
            hidden_dim_c (int, optional): hidden dimension of cnn.
            hidden_dim_skip (int, optional): hidden dimension of skip rnn.
            kernel (int, optional): kernel size of cnn.
            window_size (int, optional): the length of the last end of the short term memory, at most (sequence length - kernel + 1).
            skip_window_size (int, optional): window size of skip rnn, represents middle term memory.
            dropout_prob (float, optional): dropout prob.
        """
        super().__init__()
        self.hidden_dim_c = hidden_dim_c
        self.skip_window_size = skip_window_size
        self.p = (window_size - kernel + 1) // skip_window_size

        self.rnn = nn.GRU(hidden_dim_c, hidden_dim_skip)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            c (torch.Tensor): convolution feature. shape is (b, hc, ?)
            r (torch.Tensor): recurrent feature. shape is (b, hr)

        Returns:
            torch.Tensor: recurrent feature and middle term memory
        """
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

    def __init__(self, hidden_dim_c: int, hidden_dim_r: int):
        """Temporal attention

        Args:
            hidden_dim_c (int): _description_
            hidden_dim_r (int): _description_
        """
        assert (
            hidden_dim_c == hidden_dim_r
        ), "hidden_dim_c has to be the same as hidden_dim_r"
        super().__init__()

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            c (torch.Tensor): convolution feature. shape is (b, hc, ?)
            r (torch.Tensor): recurrent feature. shape is (b, hr)

        Returns:
            torch.Tensor: attended convolution feature
        """
        # B, 1, hr x B, hc, ? => B, 1, ?
        a = torch.bmm(r[:, None, :], c)
        # B, hc, ? x B, ?, 1 => B, hc
        y = torch.bmm(c, a.softmax(-1).transpose(-1, -2)).squeeze(-1)

        # B, hc
        return y


class Highway(nn.Module):

    def __init__(self, highway_window_size: int):
        """Highway of LSTNet in [official repo](https://github.com/laiguokun/LSTNet/blob/master/models/LSTNet.py)
        Add a sequence remixing global signal to output

        Args:
            highway_window_size (int, optional): the length of the last end of the input.
        """
        super().__init__()
        self.highway_window_size = highway_window_size
        self.linear = nn.Linear(highway_window_size, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input. shape is (b, 1, s, c)
            z (torch.Tensor): output. shape is (b, c)

        Returns:
            torch.Tensor: output
        """
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
