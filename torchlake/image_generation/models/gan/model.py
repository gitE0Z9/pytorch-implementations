from math import prod

import torch
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
        self.foot = nn.Linear(input_channel, self.hidden_dim)

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, output_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        y = y.view(-1, *self.image_shape)
        return y


class GANDiscriminator(ModelBase):
    def __init__(
        self,
        hidden_dim: int,
        image_shape: tuple[int],
        dropout_prob: float = 0.5,
    ):
        """_summary_

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
            nn.Linear(input_channel, self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        y = x.view(batch_size, -1)
        return super().forward(y)
