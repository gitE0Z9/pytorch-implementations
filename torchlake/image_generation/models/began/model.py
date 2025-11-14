from math import prod
from torch import nn
import torch.nn.functional as F
import torch
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

# TODO: vanishing residual


class BEGANGenerator(ModelBase):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dim: int,
        init_shape: tuple[int, int],
        enable_skip_connection: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.init_shape = init_shape
        self.enable_skip_connection = enable_skip_connection
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        init_shape = (self.hidden_dim, *self.init_shape)
        self.foot = nn.Sequential(
            nn.Linear(input_channel, prod(init_shape)),
            nn.Unflatten(1, init_shape),
            Conv2dNormActivation(
                self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            nn.Upsample(scale_factor=2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            Conv2dNormActivation(
                2 * self.hidden_dim if self.enable_skip_connection else self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            nn.Upsample(scale_factor=2),
            Conv2dNormActivation(
                2 * self.hidden_dim if self.enable_skip_connection else self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_size, 3, padding=1),
        )


class BEGANDiscriminator(ModelBase):

    def __init__(
        self,
        input_channel: int,
        latent_dim: int,
        hidden_dim: int,
        image_shape: tuple[int, int],
        enable_skip_connection: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.enable_skip_connection = enable_skip_connection
        super().__init__(input_channel, input_channel)

    @property
    def feature_dim(self) -> int:
        return prod(s // 4 for s in self.image_shape) * 3 * self.hidden_dim

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                self.hidden_dim,
                self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                self.hidden_dim,
                2 * self.hidden_dim,
                3,
                stride=2,
                activation_layer=nn.ELU,
            ),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            Conv2dNormActivation(
                2 * self.hidden_dim,
                2 * self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                2 * self.hidden_dim,
                3 * self.hidden_dim,
                3,
                stride=2,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                3 * self.hidden_dim,
                3 * self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
            Conv2dNormActivation(
                3 * self.hidden_dim,
                3 * self.hidden_dim,
                3,
                activation_layer=nn.ELU,
            ),
        )

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.latent_dim),
        )

    def build_head(self, output_size, **kwargs):
        self.head = BEGANGenerator(
            self.latent_dim,
            output_size,
            self.hidden_dim,
            (s // 4 for s in self.image_shape),
            enable_skip_connection=self.enable_skip_connection,
        )

    def forward(self, x: torch.Tensor, output_latent: bool = False) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y)
        y = self.neck(y)

        if output_latent:
            z = y

        # for sharpness
        if self.enable_skip_connection:
            h = self.head.foot[:2](y)
            y = self.head.foot[2:](h)  # 2x
            y = torch.cat((y, F.interpolate(h, size=y.shape[2:])), 1)
            y = self.head.blocks[:3](y)  # 4x
            y = torch.cat((y, F.interpolate(h, size=y.shape[2:])), 1)
            y = self.head.blocks[3:](y)
            y = self.head.head(y)

        if output_latent:
            return y, z

        return y
