import torch
from torch import nn

from ..utils.numerical import generate_grid


class PositionEncoding(nn.Module):

    def __init__(
        self,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.trainable = trainable

        if trainable:
            self.embedding = nn.Parameter(self.init_fourier_grid(seq_len, hidden_dim))

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
        if not self.trainable:
            _, seq_len, hidden_dim = x.shape
            return self.init_fourier_grid(seq_len, hidden_dim)

        if self.trainable:
            # TODO: interpolation
            return self.embedding
