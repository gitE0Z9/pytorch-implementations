import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        num_layers: int,
        dropout_prob: float = 0,
    ):
        self.dropout_prob = dropout_prob
        super().__init__()
        self.pool = nn.MaxUnpool2d(2, 2)
        self.blocks = nn.Sequential(
            *tuple(
                Conv2dNormActivation(
                    input_channel,
                    input_channel if i < num_layers - 1 else output_channel,
                )
                for i in range(num_layers)
            )
        )

        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, pooling_indice: torch.Tensor) -> torch.Tensor:
        y = self.pool(x, pooling_indice)
        y = self.blocks(y)

        if self.dropout_prob > 0:
            y = self.dropout(y)

        return y
