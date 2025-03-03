from math import prod

from torch import nn
from torchlake.common.models.model_base import ModelBase


class GANGenerator(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        image_shape: tuple[int],
    ):
        """Generator of GAN

        Args:
            input_channel (int): input channel size
            hidden_dim (int): dimension of hidden layer
            image_shape (tuple[int]): shape of original image
        """
        self.hidden_dim = hidden_dim
        self.image_shape = image_shape
        super().__init__(input_channel, prod(image_shape))

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            nn.Linear(input_channel, self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.LeakyReLU(0.2),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, output_size),
            nn.Tanh(),
            nn.Unflatten(-1, self.image_shape),
        )


class GANDiscriminator(ModelBase):
    def __init__(
        self,
        hidden_dim: int,
        image_shape: tuple[int],
        dropout_prob: float = 0.5,
    ):
        """Discriminator of GAN

        Args:
            hidden_dim (int): dimension of hidden layer
            image_shape (tuple[int]): shape of generated image
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
        """
        self.hidden_dim = hidden_dim
        self.image_shape = image_shape
        self.dropout_prob = dropout_prob
        super().__init__(prod(image_shape), 1)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channel, self.hidden_dim * 2),
            # nerf
            nn.LeakyReLU(0.2, inplace=True),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            # nerf
            nn.LeakyReLU(0.2, inplace=True),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.hidden_dim, output_size),
        )
