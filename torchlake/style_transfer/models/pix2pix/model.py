from itertools import pairwise

import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.models import ConvINReLU

from .network import DownSampling, UpSampling


def init_conv(layer: nn.Conv2d):
    nn.init.normal_(layer.weight, 0, 0.02)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_norm(layer: nn.BatchNorm2d):
    if layer.weight is not None:
        nn.init.normal_(layer.weight, 1, 0.02)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Pix2PixDiscriminator(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int = 64,
        num_layer: int = 3,
    ):
        """Pix2pix discriminator, a PatchGAN

        Args:
            input_channel (int): input channel size
            hidden_dim (int): dimension of hidden layers
            num_layer (int): number of blocks
        """
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        super().__init__(input_channel, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init_conv(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                init_norm(layer)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            ConvINReLU(input_channel, self.hidden_dim, 4, 2, 1),
        )

    def build_blocks(self, **kwargs):
        blocks = nn.Sequential()

        d = self.hidden_dim
        for _ in range(self.num_layer - 1):
            block = ConvINReLU(d, d * 2, 4, 2, 1)
            d *= 2
            blocks.append(block)

        block = ConvINReLU(d, d * 2, 4, 1, 1)
        blocks.append(block)

        self.blocks = blocks

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim * (2**self.num_layer), output_size, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = torch.cat([x, z], dim=1)
        y = self.foot(y)
        y = self.blocks(y)
        return self.head(y)


class Pix2PixGenerator(ModelBase):
    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dim: int = 64,
        num_block: int = 6,
        dropout_prob: float = 0.5,
    ):
        """Pix2pix generator, a UNet

        Args:
            input_channel (int): input channel size
            output_size (int): output size
            hidden_dim (int): dimension of hidden layers. Defaults to 64.
            num_block (int): number of downsampling and upsampling blocks. Defaults to 6.
        """
        self.hidden_dim = hidden_dim
        self.num_block = num_block
        self.dropout_prob = dropout_prob
        super().__init__(input_channel, output_size)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                init_conv(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                init_norm(layer)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            nn.Conv2d(input_channel, self.hidden_dim, 4, 2, 1),
        )

    def build_blocks(self, **kwargs):
        blocks = nn.Sequential()

        d = self.hidden_dim
        for _ in range(self.num_block):
            d_prime = min(d * 2, self.hidden_dim * 8)
            block = DownSampling(d, d_prime)
            d = d_prime
            blocks.append(block)
        blocks.append(DownSampling(d, d, enable_in=False))

        self.blocks = blocks

    def build_neck(self, **kwargs):
        neck = nn.ModuleList()

        input_channels = [block.output_channel for block in self.blocks][::-1]
        input_channels.append(self.hidden_dim)

        in_c, out_c = input_channels[:2]
        block = UpSampling(
            in_c,
            out_c,
            dropout_prob=self.dropout_prob,
        )
        in_c = in_c + out_c
        neck.append(block)
        for i, out_c in enumerate(input_channels[2:]):
            block = UpSampling(
                in_c,
                out_c,
                dropout_prob=self.dropout_prob if i < 2 else 0,
            )
            in_c = 2 * out_c
            neck.append(block)

        self.neck = neck

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(self.hidden_dim * 2, output_size, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)

        features = [y]
        for block in self.blocks[:-1]:
            y = block(y)
            features.append(y)

        y = self.blocks[-1](y)
        for neck in self.neck:
            y = neck(y, features.pop())

        return self.head(y)
