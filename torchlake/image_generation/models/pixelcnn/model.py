from torch import nn
from torchlake.common.models import ResBlock
from torchlake.common.models.model_base import ModelBase

from .network import BottleNeck, MaskedConv2d


class PixelCNN(ModelBase):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dim: int,
        num_layer: int,
    ):
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            MaskedConv2d(
                input_channel,
                2 * self.hidden_dim,
                7,
                mask_type="A",
                padding=3,
            ),
        )

    def build_blocks(self, **kwargs):
        self.blocks = self.num_layer * nn.Sequential(
            ResBlock(
                2 * self.hidden_dim,
                2 * self.hidden_dim,
                block=BottleNeck(self.hidden_dim),
            )
        )

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(
                2 * self.hidden_dim,
                self.hidden_dim,
                1,
                mask_type="B",
            ),
            nn.ReLU(),
            MaskedConv2d(
                self.hidden_dim,
                self.hidden_dim,
                1,
                mask_type="B",
            ),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_size, 1),
        )
