from torch import nn

from torchlake.common.models import ConvINReLU
from torchlake.common.models.imagenet_normalization import ImageNetNormalization
from torchlake.common.models.model_base import ModelBase

from .network import ResidualBlock


class FastStyleTransfer(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int = 32,
        num_block: int = 5,
    ):
        self.hidden_dim = hidden_dim
        self.num_block = num_block
        super().__init__(input_channel, input_channel)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            ImageNetNormalization(),
            ConvINReLU(input_channel, self.hidden_dim, 9),
            ConvINReLU(self.hidden_dim, self.hidden_dim * 2, 3, stride=2),
            ConvINReLU(self.hidden_dim * 2, self.hidden_dim * 4, 3, stride=2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            *tuple(ResidualBlock(self.hidden_dim * 4) for _ in range(self.num_block))
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvINReLU(self.hidden_dim * 4, self.hidden_dim * 2, 3),
            nn.Upsample(scale_factor=2),
            ConvINReLU(self.hidden_dim * 2, self.hidden_dim, 3),
            ConvINReLU(
                self.hidden_dim,
                output_size,
                9,
                enable_in=False,
                activation=None,
            ),
            # nn.Tanh(),
        )
