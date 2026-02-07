import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import RCU, RefineNetBlock


class RefineNet(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 256,
    ):
        """RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation [1611.06612v3]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            hidden_dim (int, optional): hidden dimension of the decoder. Defaults to 256.
        """
        self.hidden_dim = hidden_dim
        super().__init__(
            1,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            RefineNetBlock(self.foot.hidden_dim_32x, 0, self.hidden_dim * 2),
            RefineNetBlock(
                self.foot.hidden_dim_16x,
                self.hidden_dim * 2,
                self.hidden_dim,
            ),
            RefineNetBlock(
                self.foot.hidden_dim_8x,
                self.hidden_dim,
                self.hidden_dim,
            ),
            RefineNetBlock(
                self.foot.hidden_dim_4x,
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            RCU(self.hidden_dim),
            RCU(self.hidden_dim),
            nn.Conv2d(self.hidden_dim, output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = self.foot(x)

        y = self.blocks[0](features.pop())
        for block in self.blocks[1:]:
            y = block(features.pop(), y)

        y = self.head(y)
        cropper = CenterCrop(x.shape[2:])
        y = F.interpolate(
            y,
            mode="bilinear",
            scale_factor=4,
            align_corners=True,
        )
        y = cropper(y)

        return y
