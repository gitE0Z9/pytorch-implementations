import torch
from torch import nn
from torchlake.common.models.model_base import ModelBase


class AutoEncoder(ModelBase):
    def __init__(
        self,
        input_channel: int,
        latent_dim: int,
        activation: nn.Module,
    ):
        self.latent_dim = latent_dim
        super().__init__(input_channel, input_channel)
        self.activation = activation

    def add_stack(self):
        l = len(self.foot)
        in_c, out_c = self.latent_dim >> (l - 1), self.latent_dim >> l
        self.foot.insert(l, nn.Linear(in_c, out_c))
        self.head.insert(0, nn.Linear(out_c, in_c))

    def del_stack(self):
        self.foot.pop(-1)
        self.head.pop(0)

    def build_foot(self, input_channel: int, **kwargs):
        self.foot = nn.Sequential(
            nn.Linear(input_channel, self.latent_dim),
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, output_size),
        )

    def forward(self, x: torch.Tensor, output_latent: bool = False) -> torch.Tensor:
        original_shape = x.shape
        b = x.size(0)
        x = x.view(b, -1)

        z = self.foot(x)
        if self.activation is not None:
            z = self.activation(z)
        y = self.head(z)

        y = y.view(*original_shape)

        if output_latent:
            return y, z

        return y
