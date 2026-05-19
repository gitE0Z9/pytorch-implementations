import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models.conv import ConvINReLU
from torchlake.common.models.model_base import ModelBase


class TextureNet(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 3,
        hidden_dim: int = 8,
        num_scale_factor: int = 5,
    ):
        self.hidden_dim = hidden_dim
        self.num_scale_factor = num_scale_factor
        super().__init__(input_channel, output_size)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.xavier_uniform_(module.bias)

    def build_foot(self, input_channel: int, **kwargs):
        # upsampling
        self.foot = nn.ModuleList(
            [
                nn.Sequential(
                    ConvINReLU(input_channel, self.hidden_dim, 3),
                    ConvINReLU(self.hidden_dim, self.hidden_dim, 3),
                    ConvINReLU(self.hidden_dim, self.hidden_dim, 1, activation=None),
                )
                for _ in range(self.num_scale_factor)
            ]
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(self.hidden_dim * l),
                    nn.BatchNorm2d(self.hidden_dim * l),
                    nn.BatchNorm2d(self.hidden_dim * l),
                    nn.BatchNorm2d(self.hidden_dim * l),
                )
                for l in range(1, self.num_scale_factor)
            ]
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                nn.Sequential(
                    ConvINReLU(self.hidden_dim * l, self.hidden_dim * l, 3),
                    ConvINReLU(self.hidden_dim * l, self.hidden_dim * l, 3),
                    ConvINReLU(self.hidden_dim * l, self.hidden_dim * l, 1),
                )
                for l in range(2, self.num_scale_factor + 1)
            ]
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            ConvINReLU(self.hidden_dim * self.num_scale_factor, output_size, 1),
        )

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        y = self.foot[0](xs.pop())
        for i in range(1, self.num_scale_factor):
            z = self.foot[i](xs.pop())
            y = F.interpolate(y, scale_factor=2)
            y = self.blocks[i - 1](y)
            y = torch.cat((y, z), 1)
            y = self.neck[i - 1](y)

        return self.head(y)
