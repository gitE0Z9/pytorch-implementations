from math import prod
from typing import Sequence

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import ConvBNReLU
from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.model_base import ModelBase


class DCGANGenerator(ModelBase):
    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dim: int = 1024,
        num_block: int = 4,
        init_shape: Sequence[int] = (4, 4),
    ):
        self.hidden_dim = hidden_dim
        self.num_block = num_block

        self.init_shape = init_shape
        self.final_shape = tuple(s * (2**num_block) for s in init_shape)

        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel):
        self.foot = nn.Sequential(
            nn.Linear(input_channel, self.hidden_dim * prod(self.init_shape)),
            nn.Unflatten(-1, (self.hidden_dim, *self.init_shape)),
            nn.BatchNorm2d(self.hidden_dim),
        )
        nn.init.normal_(self.foot[2].weight, 1, 0.02)
        nn.init.constant_(self.foot[2].bias, 0)

    def build_blocks(self):
        blocks = []
        for i in range(self.num_block):
            blocks.append(
                ConvBNReLU(
                    self.hidden_dim // (2**i),
                    self.hidden_dim // (2 ** (i + 1)),
                    4,
                    padding=1,
                    stride=2,
                    activation=nn.ReLU(),
                    deconvolution=True,
                ),
            )

        self.blocks = nn.Sequential(*blocks)

        nn.init.normal_(self.blocks[1].conv.weight, 0, 0.02)
        self.blocks[1].bn.momentum = 0.8
        nn.init.normal_(self.blocks[1].bn.weight, 1, 0.02)
        nn.init.constant_(self.blocks[1].bn.bias, 0)
        nn.init.normal_(self.blocks[3].conv.weight, 0, 0.02)
        self.blocks[3].bn.momentum = 0.8
        nn.init.normal_(self.blocks[3].bn.weight, 1, 0.02)
        nn.init.constant_(self.blocks[3].bn.bias, 0)

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            # compare to conv2d + tanh, still has reflection pad
            ConvBNReLU(
                self.hidden_dim // (2**self.num_block),
                output_size,
                3,
                padding=1,
                enable_bn=True,
                activation=nn.Tanh(),
                deconvolution=True,
            ),
        )
        nn.init.normal_(self.head[0].conv.weight, 0, 0.02)


class DCGANDiscriminator(ModelBase):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        image_shape: Sequence[int],
        num_block: int,
    ):
        self.hidden_dim = hidden_dim
        self.num_block = num_block

        assert len(image_shape) == 2, "image size must be (height, width)"
        for size in image_shape:
            assert (
                size % (2**num_block) == 0
            ), "image size must be divisible by 2^num_block"

        self.init_image_shape = image_shape
        self.final_image_shape = tuple(
            size // (2 ** (self.num_block + 1)) for size in self.init_image_shape
        )
        super().__init__(input_channel, 1)

    @property
    def feature_dim(self) -> int:
        return prod(self.final_image_shape) * self.hidden_dim * (2**self.num_block)

    def build_foot(self, input_channel: int):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.hidden_dim,
                3,
                stride=2,
                norm_layer=None,
                activation_layer=lambda: nn.LeakyReLU(0.2),
                inplace=None,
            ),
        )

        # self.foot[0][1].momentum = 0.8

    def build_blocks(self):
        blocks = []
        for i in range(self.num_block):
            blocks.append(
                Conv2dNormActivation(
                    self.hidden_dim * (2**i),
                    self.hidden_dim * (2 ** (i + 1)),
                    3,
                    stride=2,
                    activation_layer=lambda: nn.LeakyReLU(0.2),
                    inplace=None,
                ),
            )
            # blocks[-1][1].momentum = 0.8

        self.blocks = nn.Sequential(*blocks)

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            FlattenFeature(reduction=None),
            nn.Linear(self.feature_dim, output_size),
        )
