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
        self.mask_groups = input_channel
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self._h = hidden_dim * self.mask_groups
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            MaskedConv2d(
                input_channel,
                2 * self._h,
                7,
                mask_type="A",
                padding=3,
                mask_groups=self.mask_groups,
            ),
        )

    def build_blocks(self, **kwargs):
        self.blocks = self.num_layer * nn.Sequential(
            ResBlock(
                2 * self._h,
                2 * self._h,
                block=BottleNeck(
                    self._h,
                    mask_groups=self.mask_groups,
                ),
            )
        )

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(
                2 * self._h,
                self._h,
                1,
                mask_type="B",
                mask_groups=self.mask_groups,
            ),
            nn.ReLU(),
            MaskedConv2d(
                self._h,
                self._h,
                1,
                mask_type="B",
                mask_groups=self.mask_groups,
            ),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            # don't add relu here
            MaskedConv2d(
                self._h,
                self.mask_groups * output_size,
                1,
                mask_type="B",
                mask_groups=self.mask_groups,
            ),
            nn.Unflatten(1, (self.mask_groups, output_size)),
        )
