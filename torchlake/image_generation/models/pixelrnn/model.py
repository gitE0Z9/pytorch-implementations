from typing import Literal

import torch
from torch import nn
from torchlake.common.models import ResBlock
from torchlake.common.models.model_base import ModelBase

from ..pixelcnn.network import MaskedConv2d
from .network import BottleNeck


class PixelRNN(ModelBase):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dim: int,
        num_layer: int,
        rnn_type: Literal["row", "diag"],
    ):
        self.mask_groups = input_channel
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self._h = hidden_dim * input_channel
        self.rnn_type = rnn_type
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            MaskedConv2d(input_channel, 2 * self._h, 7, mask_type="A", padding=3),
        )

    def build_blocks(self, **kwargs):
        self.blocks = self.num_layer * nn.Sequential(
            ResBlock(
                2 * self._h,
                2 * self._h,
                block=BottleNeck(
                    self._h,
                    2 if self.rnn_type == "diag" else 3,
                    type=self.rnn_type,
                ),
            )
        )

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(2 * self._h, self._h, 1, mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(self._h, self._h, 1, mask_type="B"),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            # don't add relu here
            MaskedConv2d(self._h, self.mask_groups * output_size, 1, mask_type="B"),
            nn.Unflatten(1, (self.mask_groups, output_size)),
        )
