import math
import torch
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import init_conv_to_zeros, init_deconv_with_bilinear_kernel


class FCN(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        output_stride: int = 8,
    ):
        """Fully Convolutional Networks for Semantic Segmentation in paper [1605.06211v1]

        Args:
            backbone (ExtractorBase): backbone.
            output_size (int, optional): output size. Defaults to 1.
            output_stride (int, optional): output stride, e.g. 4 means FCN-4s, 8 means FCN-8s, 16 means FCN-16s, 32 means FCN-32s and so on. Defaults to 8.
        """
        self.output_stride = output_stride
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    @property
    def num_skip_connection(self) -> int:
        return 5 - int(math.log2(self.output_stride))

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList()

        # from high to low
        # i.e. stage5, stage4, stage3, stage2, stage1
        dims = [self.foot.feature_dim]
        s = 16
        while self.output_stride <= s:
            dims.append(getattr(self.foot, f"hidden_dim_{s}x"))
            s //= 2

        for dim in dims:
            layer = nn.Conv2d(dim, self.output_size, 1)
            init_conv_to_zeros(layer)
            self.blocks.append(layer)

    def build_head(self, output_size, **kwargs):
        # 8s => k=16, s=8
        # 16s => k=32, s=16
        # 32s => k=64, s=32

        # from high to low
        # except the first one is lowest
        # i.e. final upsampling, stage5, stage4, stage3, stage2, stage1
        self.head = nn.ModuleList()

        for _ in range(self.num_skip_connection):
            layer = nn.ConvTranspose2d(
                output_size,
                self.output_size,
                4,
                stride=2,
                bias=False,
            )
            init_deconv_with_bilinear_kernel(layer)
            self.head.append(layer)

        layer = nn.ConvTranspose2d(
            output_size,
            output_size,
            self.output_stride * 2,
            stride=self.output_stride,
            bias=False,
        )
        init_deconv_with_bilinear_kernel(layer)
        layer.requires_grad_(False)
        self.head.insert(0, layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # vgg
        # 8s => 3, 4, 6
        # 16s => 4, 6
        # 32s => 6
        features: list[torch.Tensor] = self.foot(x)

        y = self.blocks[0](features.pop())
        for head, block in zip(self.head[1:], self.blocks[1:]):
            # upsampling low-level features
            y = head(y)

            # feature fusion
            cropper = CenterCrop(y.shape[-2:])
            y = y + cropper(block(features.pop()))

        y = self.head[0](y)
        return CenterCrop(x.shape[-2:])(y)
