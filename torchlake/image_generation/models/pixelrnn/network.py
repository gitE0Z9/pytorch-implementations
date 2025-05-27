from typing import Literal
import torch
from torch import nn

from ..pixelcnn.network import MaskedConv2d


class RowLSTM(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        kernel: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # is: input to state
        self.conv_is = MaskedConv2d(
            input_channel,
            4 * hidden_dim,
            kernel=(1, kernel),
            mask_type="B",
            padding=(0, kernel // 2),
        )
        # ss: state to state
        self.conv_ss = nn.Conv2d(
            hidden_dim,
            4 * hidden_dim,
            kernel_size=(1, kernel),
            padding=(0, kernel // 2),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        B, _, H, W = x.size()
        h = torch.zeros(B, self.hidden_dim, 1, W, device=x.device)
        c = torch.zeros(B, self.hidden_dim, 1, W, device=x.device)

        z_is = self.conv_is(x)
        outputs = []
        for i in range(H):
            # use conv to get row level representation
            z = z_is[:, :, i : i + 1, :] + self.conv_ss(h)
            i, f, o = z[:, : 3 * self.hidden_dim, :, :].sigmoid().chunk(3, 1)
            g = z[:, -self.hidden_dim :, :, :].tanh()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h)
        return torch.cat(outputs, dim=2)


class DiagonalLSTM(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        kernel: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel

        # is: input to state
        self.conv_is = MaskedConv2d(
            input_channel,
            4 * hidden_dim,
            kernel=1,
            mask_type="B",
        )
        # ss: state to state
        self.conv_ss = nn.Conv2d(
            hidden_dim,
            4 * hidden_dim,
            kernel_size=(kernel, 1),
            padding=(kernel // 2, 0),
        )

    @staticmethod
    def skew(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        y = torch.zeros(B, C, H, 2 * W - 1).to(x.device)

        row_indices = torch.arange(H, device=x.device).view(-1, 1)
        col_indices = torch.arange(W, device=x.device).view(1, -1) + row_indices
        y[:, :, row_indices, col_indices] = x

        return y

    @staticmethod
    def unskew(x: torch.Tensor) -> torch.Tensor:
        _, _, H, W_PRIME = x.size()
        W = (W_PRIME + 1) // 2

        row_indices = torch.arange(H, device=x.device).view(-1, 1)
        col_indices = torch.arange(W, device=x.device).view(1, -1) + row_indices

        return x[:, :, row_indices, col_indices]

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        B, _, H, W = x.size()
        h = torch.zeros(B, self.hidden_dim, H, 1, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H, 1, device=x.device)

        z_is = self.conv_is(self.skew(x))
        outputs = []
        for j in range(z_is.size(3)):
            # diagonal
            z = (
                z_is[:, :, :, j : j + 1]
                + self.conv_ss(h)[:, :, : -(self.kernel - 1), :]
            )
            # lstm
            i, f, o = z[:, : 3 * self.hidden_dim, :, :].sigmoid().chunk(3, 1)
            g = z[:, -self.hidden_dim :, :, :].tanh()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h)
        outputs = torch.cat(outputs, dim=3)
        outputs = self.unskew(outputs)

        return outputs


class DiagonalBiLSTM(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        kernel: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel

        self.lstm_l = DiagonalLSTM(input_channel, hidden_dim, kernel)
        self.lstm_r = DiagonalLSTM(input_channel, hidden_dim, kernel)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        zl = self.lstm_l(x, h)
        zr = self.lstm_r(x.flip(2), h)

        zl[:, :, 1:, :] = zl[:, :, 1:, :] + zr[:, :, :-1, :]

        return zl


class BottleNeck(nn.Sequential):
    def __init__(
        self,
        hidden_dim: int,
        kernel: int,
        type: Literal["row", "diag"],
    ):
        super().__init__(
            (
                RowLSTM(2 * hidden_dim, hidden_dim, kernel)
                if type == "row"
                else DiagonalBiLSTM(2 * hidden_dim, hidden_dim, kernel)
            ),
            MaskedConv2d(hidden_dim, 2 * hidden_dim, 1, mask_type="B"),
        )
