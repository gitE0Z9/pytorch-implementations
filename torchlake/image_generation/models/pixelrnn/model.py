from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models import ResBlock
from torchlake.common.models.encoder_decoder_base import EncoderDecoderModel
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
        bidirectional: bool = False,
    ):
        self.mask_groups = input_channel
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self._h = hidden_dim * self.mask_groups
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            MaskedConv2d(
                input_channel,
                2 * self._h,
                7,
                mask_type="A",
                padding=3,
                mask_groups=self.mask_groups,
            ),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            *[
                ResBlock(
                    2 * self._h,
                    2 * self._h,
                    block=BottleNeck(
                        self._h,
                        2 if self.rnn_type == "diag" else 3,
                        type=self.rnn_type,
                        bidirectional=self.bidirectional,
                    ),
                )
                for _ in range(self.num_layer)
            ]
        )

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(
                2 * self._h,
                self._h,
                1,
                mask_type="B",
                mask_groups=self.mask_groups,
            ),
            nn.ReLU(),
            MaskedConv2d(
                self._h,
                self._h,
                1,
                mask_type="B",
                mask_groups=self.mask_groups,
            ),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            # don't add ReLU before this MaskedConv2d layer
            MaskedConv2d(
                self._h,
                self.mask_groups * output_size,
                1,
                mask_type="B",
                mask_groups=self.mask_groups,
            ),
            nn.Unflatten(1, (self.mask_groups, output_size)),
        )


class MultiScalePixelRNN(EncoderDecoderModel[PixelRNN, PixelRNN]):
    def __init__(
        self,
        encoder: PixelRNN,
        decoder: PixelRNN,
        decoder_kernel: int = 3,
        scale_factor: int = 4,
    ):
        """Multi-scale PixelRNN, a PixelRNN with large image conditioned on a PixelRNN with small image

        Args:
            input_channel (int): input channel size
            hidden_dim (int): dimension of hidden layers
            decoder_kernel (int): kernel size of the deconvolution layer
            scale_factor (int): scale for unconditional PixelRNN
        """
        self.decoder_kernel = decoder_kernel
        self.scale_factor = scale_factor
        super().__init__(encoder, decoder)

    def build_neck(self, **kwargs):
        # self.neck = nn.ConvTranspose2d(
        #     self.encoder.output_size,
        #     self.decoder.hidden_dim,
        #     self.decoder_kernel,
        # )
        self.neck = nn.Upsample(scale_factor=self.scale_factor)

    def build_branch(self, **kwargs):
        h = self.decoder.hidden_dim
        self.branch = nn.ModuleList(
            [
                nn.Conv2d(self.encoder.output_size, 4 * h, 1)
                for _ in range(self.decoder.num_layer)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.interpolate(x, scale_factor=1 / self.scale_factor)
        z = self.encoder(z)
        z = self.neck(z)

        y = self.decoder.foot(x)
        for block, bridge in zip(self.decoder.blocks, self.branch):
            # block is ResBlock
            # pass conditional tensor through residual block
            y = block.block(y, bridge(z)) + block.downsample(y)
            if block.activation is not None:
                y = block.activation(y)

        y = self.decoder.neck(y)
        return self.decoder.head(y)
