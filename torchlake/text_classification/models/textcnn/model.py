import torch
from torch import nn

from .network import TextCnnPool


class TextCnn(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        label_size: int,
        padding_idx: int | None = None,
        kernel_size: list[int] = [2, 3, 4],
    ):
        super(TextCnn, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pool = TextCnnPool(embed_dim, kernel_size)
        self.fc = nn.Linear(32 * len(kernel_size), label_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch_size, 1, Seq_len, embed_dim
        y = self.embed(x).unsqueeze(1)

        # Batch_size, filter_number
        y = self.pool(y)

        y = self.fc(y)

        return y
