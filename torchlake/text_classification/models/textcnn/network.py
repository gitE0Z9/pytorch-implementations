import torch
from torch import nn
from torch.nn import functional as F


class TextCnnPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_size: list[int] = [2, 3, 4],
    ):
        super(TextCnnPool, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 32, (kernel, embed_dim)),
                    nn.LeakyReLU(),
                )
                for kernel in kernel_size
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch_size, filter_number, Width
        ys = [conv(x).squeeze(3) for conv in self.convs]

        # Batch_size, filter_number
        ys = [F.max_pool1d(y, y.size(2)).squeeze(2) for y in ys]

        y = torch.cat(ys, 1)

        return y
