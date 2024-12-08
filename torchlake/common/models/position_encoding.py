import torch
from torch import nn
import torch.nn.functional as F

from ..utils.numerical import generate_grid


class PositionEncoding1d(nn.Module):

    def __init__(
        self,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
        trainable: bool = False,
    ):
        """Position encoding for 1d sequence

        Args:
            seq_len (int | None, optional): sequence length for fixed size. Defaults to None.
            hidden_dim (int | None, optional): hidden dimension for fixed size. Defaults to None.
            trainable (bool, optional): is encoding parameters fixed size and trainable parameters. Defaults to False.
        """
        super().__init__()
        self.trainable = trainable

        if trainable:
            self.encoding = nn.Parameter(self.init_fourier_grid(seq_len, hidden_dim))

    def init_fourier_grid(
        self,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
    ) -> torch.Tensor:
        grid_x, grid_y = generate_grid(hidden_dim, seq_len)
        # TODO: if they truly share freq, just add pi/2 for sin to cos
        grid_x = grid_x[:, 0::2].repeat_interleave(2, 1)
        denominator = 10000 ** (2 * grid_x / hidden_dim)
        y = grid_y / denominator

        y[:, 0::2] = y[:, 0::2].sin()
        y[:, 1::2] = y[:, 1::2].cos()

        # 1, S, h
        return y.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """generate position encoding
        if trainable: x should be channel first, return interpolated channel first fourier grid
        if not trainable: x should have shape (b, s, h). return fourier grid in same shape

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: fourier grid
        """
        if self.trainable:
            return F.interpolate(self.encoding.permute(0, 2, 1), size=x.shape[2:])
        else:
            _, seq_len, hidden_dim = x.shape
            return self.init_fourier_grid(seq_len, hidden_dim)
