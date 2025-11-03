import torch
from torch import nn

from torchlake.common.models import ConvBNReLU
from torchlake.common.models.model_base import ModelBase

from ...utils.initialization import init_conv_ebgan_style


# init W ~ N(0, 0.002) W0 ~ 0
# first layer no BN
class EBGANDiscriminator(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        hidden_dim: int = 64,
        num_layer: int = 3,
    ):
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        super().__init__(input_channel, input_channel)

    def build_foot(self, input_channel, **kwargs):
        blocks = []
        for i in range(self.num_layer):
            block = ConvBNReLU(
                input_channel if i == 0 else self.hidden_dim * (2 ** (i - 1)),
                self.hidden_dim * (2**i),
                4,
                padding=1,
                stride=2,
                enable_bn=i != 0,
            )
            init_conv_ebgan_style(block.conv)
            blocks.append(block)
        self.foot = nn.Sequential(*blocks)

    def build_head(self, output_size, **kwargs):
        blocks = []
        for i in range(self.num_layer - 1, -1, -1):
            block = ConvBNReLU(
                self.hidden_dim * (2**i),
                output_size if i == 0 else self.hidden_dim * (2 ** (i - 1)),
                4,
                padding=1,
                stride=2,
                deconvolution=True,
                activation=None if i == 0 else nn.ReLU(),
            )
            init_conv_ebgan_style(block.conv)
            blocks.append(block)
        self.head = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, output_latent: bool = False) -> torch.Tensor:
        z = self.foot(x)
        y = self.head(z)

        if output_latent:
            return y, z.view(z.size(0), -1)
        else:
            return y
