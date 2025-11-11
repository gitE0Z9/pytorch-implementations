import torch
from torch import nn
from torchlake.common.models.model_base import ModelBase


class AutoEncoder(ModelBase):
    def __init__(self, input_channel: int, latent_dim: int):
        self.latent_dim = latent_dim
        super().__init__(input_channel, input_channel)

    def build_foot(self, input_channel: int, **kwargs):
        self.foot = nn.Sequential(
            nn.Linear(input_channel, self.latent_dim),
            nn.Sigmoid(),
        )

    def build_head(self, input_channel: int, **kwargs):
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, input_channel),
        )

    def forward(self, x: torch.Tensor, output_latent: bool = False) -> torch.Tensor:
        original_shape = x.shape
        b = x.size(0)
        x = x.view(b, -1)

        z = self.foot(x)
        y = self.head(z)

        y = y.view(*original_shape)

        if output_latent:
            return y, z

        return y
