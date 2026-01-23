import torch
from torch import nn
from torchvision.transforms import CenterCrop
import torch.nn.functional as F

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import LRASPP


class MobileNetV3Seg(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 128,
        pool_kernel_size: tuple[int] = (49, 49),
        pool_stride: tuple[int] = (16, 20),
        output_stride: int = 8,
    ):
        """MobileNet v3 semantic segmentation in paper [1905.02244v5]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            hidden_dim (int): dimension of lr-aspp layer
            pool_kernel_size (tuple[int], optional): kernel size of pool. Defaults to (49, 49).
            pool_stride (tuple[int], optional): stride of pool. Defaults to (16, 20).
        """
        self.hidden_dim = hidden_dim
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.output_stride = output_stride
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(
            (
                f"2_{len(self.foot.get_stage()[-3])}",  # 8x
                f"4_{len(self.foot.get_stage()[-1])-1}",  # 32x
            )
        )

    def build_head(self, output_size: int, **kwargs) -> LRASPP:
        """lr aspp

        Args:
            output_size (int, optional): output size.
        Returns:
            LRASPP: lr aspp
        """
        self.head = nn.Sequential(
            LRASPP(
                self.foot.feature_dims[-3][-1],
                self.foot.feature_dims[-1][-2],
                upsample_scale=2,
                hidden_dim=self.hidden_dim,
                output_channel=output_size,
                pool_kernel_size=self.pool_kernel_size,
                pool_stride=self.pool_stride,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.foot(x)
        y = self.head[0](*features)

        cropper = CenterCrop(x.shape[2:])
        y = F.interpolate(
            y,
            scale_factor=self.output_stride,
            mode="bilinear",
            align_corners=True,
        )
        y = cropper(y)
        return y
