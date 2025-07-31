from typing import Sequence

import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase

from .network import (
    ConcatELU,
    DownsampleLayer,
    DownwardAndRightwardConv2d,
    DownwardConv2d,
    ResidualLayer,
    UpsampleLayer,
    shift_downward,
    shift_rightward,
)


class PixelCNNPP(ModelBase):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        num_mixture: int,
        num_stage: int = 3,
        num_block: int = 1,
        dropout_prob: float = 0.1,
        conditional_shape: Sequence[int] | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_mixture = num_mixture
        self.num_stage = num_stage
        self.num_block = num_block
        self.dropout_prob = dropout_prob
        self.conditional_shape = conditional_shape
        super().__init__(
            input_channel,
            # 1 is mixture indicator
            # 2 * input_channel is mu and logsigma
            # sum(range(input_channel)) is AR coef
            (num_mixture * (1 + 2 * input_channel + sum(range(input_channel)))),
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.ModuleDict(
            {
                # see all of above context
                "v": DownwardConv2d(
                    input_channel,
                    self.hidden_dim,
                    (2, 3),
                    padding=(0, 1),
                ),
                # see only one row above context
                "hh": DownwardConv2d(
                    input_channel,
                    self.hidden_dim,
                    (1, 3),
                    padding=(0, 1),
                ),
                # see current above context, h looks like a cross
                "hv": DownwardAndRightwardConv2d(
                    input_channel,
                    self.hidden_dim,
                    (2, 1),
                ),
            }
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList()
        for i in range(self.num_stage):
            for _ in range(self.num_block):
                self.blocks.append(
                    ResidualLayer(
                        self.hidden_dim,
                        self.hidden_dim,
                        dropout_prob=self.dropout_prob,
                        conditional_shape=self.conditional_shape,
                    )
                )
            if i < self.num_stage - 1:
                self.blocks.append(DownsampleLayer(self.hidden_dim, self.hidden_dim))

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList()
        for i in range(self.num_stage):
            for _ in range(self.num_block if i == 0 else self.num_block + 1):
                self.neck.append(
                    ResidualLayer(
                        self.hidden_dim,
                        self.hidden_dim,
                        is_upside=True,
                        dropout_prob=self.dropout_prob,
                        conditional_shape=self.conditional_shape,
                    )
                )
            if i < self.num_stage - 1:
                self.neck.append(UpsampleLayer(self.hidden_dim, self.hidden_dim))

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            ConcatELU(),
            nn.Conv2d(2 * self.hidden_dim, output_size, 1),
            nn.Unflatten(1, (self.num_mixture, output_size // self.num_mixture)),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is not None and len(cond.shape) == 2:
            cond = cond[:, :, None, None]

        v = shift_downward(self.foot["v"](x), 1)
        h = shift_rightward(self.foot["hh"](x), 1) + shift_downward(
            self.foot["hv"](x), 1
        )

        v_features = [v]
        h_features = [h]
        for layer in self.blocks:
            if isinstance(layer, DownsampleLayer):
                v, h = layer(v, h)
            else:
                v, h = layer(v, h, cond=cond)
            v_features.append(v)
            h_features.append(h)

        # drop last items, duplicate with current v, h
        v_features.pop(), h_features.pop()

        for layer in self.neck:
            if isinstance(layer, UpsampleLayer):
                v, h = layer(v, h)
            else:
                v, h = layer(v, h, v_features.pop(), h_features.pop(), cond=cond)

        return self.head(h)
