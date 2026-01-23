import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from ..deeplabv3.network import ASPP


class MobileNetV2Seg(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 256,
        output_stride: int = 8,
    ):
        """MobileNet v2 semantic segmentation in paper [1801.04381v4]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            reduction_ratio (int, optional): prediction head dimension reduced ratio. Defaults to 8.
        """
        self.hidden_dim = hidden_dim
        self.output_stride = output_stride
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

        num_layer_of_last_stage = len(self.foot.get_stage()[-1])
        self.foot.fix_target_layers((f"4_{num_layer_of_last_stage-1}",))

    def build_neck(self):
        """Reduced ASPP: an ASPP withotut 3x3

        Returns:
            nn.Module: neck module
        """
        self.neck = nn.Sequential(
            ASPP(
                # penultimate layer of last stage
                self.foot.feature_dims[-1][-2],
                self.hidden_dim,
                dilations=[],
            ),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.foot(x)
        y = self.neck(features.pop())
        y = self.head(y)

        cropper = CenterCrop(x.shape[2:])
        y = F.interpolate(
            y,
            scale_factor=self.output_stride,
            mode="bilinear",
            align_corners=True,
        )
        y = cropper(y)
        return y
