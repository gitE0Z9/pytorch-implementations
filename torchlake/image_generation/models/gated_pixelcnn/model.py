from typing import Sequence
from torch import nn
import torch
from torchlake.common.models.model_base import ModelBase

from .network import GatedLayer, MaskedConv2d


class GatedPixelCNN(ModelBase):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        hidden_dim: int,
        num_layers: int,
        conditional_shape: Sequence[int] | None = None,
    ):
        """Gated PixelCNN. [1606.05328v2]

        Args:
            input_channel (int): input channel size.
            output_size (int, optional): output size.
            hidden_dim (int): hidden dimension
            num_layers (int): number of gated layers
            conditional_shape (Sequence[int] | None, optional): the shape of the conditional representation. Default is None.
        """
        self.mask_groups = input_channel
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._h = hidden_dim * self.mask_groups
        self.conditional_shape = conditional_shape
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
        self.blocks = nn.ModuleList(
            [
                GatedLayer(
                    self._h,
                    3,
                    conditional_shape=self.conditional_shape,
                    mask_groups=self.mask_groups,
                )
                for _ in range(self.num_layers)
            ]
        )

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(
                self._h,
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

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is not None and len(cond.shape) == 2:
            cond = cond[:, :, None, None]

        y = self.foot(x)

        v, h = self.blocks[0](
            y[:, : self._h],
            y[:, -self._h :],
            cond,
        )
        for block in self.blocks[1:]:
            v, h = block(v, h, cond)

        y = self.neck(h)

        return self.head(y)
