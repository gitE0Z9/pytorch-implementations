from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.sequence import get_input_sequence
from torchlake.sequence_data.models.base import RNNDiscriminator, RNNGenerator
from typing import Type


class RNNLM(RNNGenerator):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        rnn_discriminator_class: Type[RNNDiscriminator],
        num_layers: int = 1,
        context: NLPContext | None = None,
    ):
        backbone = rnn_discriminator_class(
            input_channel=vocab_size,
            hidden_dim=hidden_dim,
            output_size=vocab_size,
            num_layers=num_layers,
            output_sequence=True,
            context=context,
            projection_layer=lambda x: F.one_hot(x, num_classes=vocab_size).float(),
        )
        super().__init__(backbone, attention=None)

    def train(self, mode=True):
        result = super().train(mode)
        # override super(RNNGenerator) behavior
        result.head.output_sequence = True
        result.head.forward = partial(result.head.forward, output_state=not mode)
        result.forward = self.head.forward if mode else self.predict
        return result

    @property
    def input_embedding(self) -> nn.Embedding:
        w: torch.Tensor = self.head.blocks.weight_ih_l0.detach()
        out_c, in_c = w.shape
        e = nn.Embedding(
            in_c,
            out_c,
            padding_idx=self.head.context.padding_idx,
        )
        e.weight.data.copy_(w.T)

        return e

    @property
    def output_embedding(self) -> nn.Embedding:
        w = self.head.head.weight.detach()
        out_c, in_c = w.shape
        e = nn.Embedding(
            out_c,
            in_c,
            padding_idx=self.head.context.padding_idx,
        )
        e.weight.data.copy_(w)

        return e
