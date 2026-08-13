from itertools import pairwise
from typing import Sequence

from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock

from .network import BottleNeck


class TCN(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int | Sequence[int],
        output_size: int,
        kernel: int | Sequence[int] = 2,
        num_block: int = 1,
        dropout_prob: float = 0.2,
    ):
        if not isinstance(hidden_dim, Sequence):
            hidden_dim = [hidden_dim] * num_block
        if not isinstance(kernel, Sequence):
            kernel = [kernel] * num_block

        assert (
            len(hidden_dim) == len(kernel) == num_block
        ), f"hidden_dim, kernel, and num_block must all agree in length (got {len(hidden_dim)}, {len(kernel)}, {num_block})"

        self.hidden_dims = hidden_dim
        self.kernels = kernel
        self.num_block = num_block
        self.dropout_prob = dropout_prob
        super().__init__(input_channel, output_size)

        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.normal_(module.bias.data)

    @property
    def feature_dim(self) -> int:
        return self.hidden_dims[-1]

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            ResBlock(
                input_channel,
                self.hidden_dims[0],
                BottleNeck(
                    input_channel,
                    self.hidden_dims[0],
                    self.kernels[0],
                    padding=self.kernels[0] - 1,
                    dropout=self.dropout_prob,
                ),
                dimension="1d",
            )
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            *[
                ResBlock(
                    prev_dim,
                    next_dim,
                    block=BottleNeck(
                        prev_dim,
                        next_dim,
                        self.kernels[i],
                        padding=(self.kernels[i] - 1) * 2**i,
                        dilation=2**i,
                        dropout=self.dropout_prob,
                    ),
                    dimension="1d",
                )
                for i, (prev_dim, next_dim) in enumerate(
                    pairwise(self.hidden_dims), start=1
                )
            ]
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            nn.Conv1d(self.feature_dim, output_size, 1),
        )
