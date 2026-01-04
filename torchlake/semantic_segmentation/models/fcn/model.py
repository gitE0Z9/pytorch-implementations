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
        num_skip_connection: int = 2,
    ):
        """Fully Convolutional Networks for Semantic Segmentation in paper [1605.06211v1]

        Args:
            backbone (ExtractorBase): backbone.
            output_size (int, optional): output size. Defaults to 1.
            num_skip_connection (int, optional): number of skip connection, e.g. 2 means FCN-8s, 1 means FCN-16s, 0 means FCN-32s. Defaults to 2.
        """
        self.num_skip_connection = num_skip_connection
        super().__init__(1, output_size, foot_kwargs={"backbone": backbone})

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        layer_names = ["6_1"]
        if self.num_skip_connection > 0:
            layer_names.append("4_1")
        if self.num_skip_connection > 1:
            layer_names.append("3_1")
        self.foot.fix_target_layers(layer_names)

    def build_blocks(self, **kwargs):
        layer = nn.Conv2d(self.foot.feature_dim, self.output_size, 1)
        init_conv_to_zeros(layer)
        self.blocks = nn.ModuleList([layer])

        if self.num_skip_connection > 0:
            layer = nn.Conv2d(self.foot.hidden_dim_16x, self.output_size, 1)
            init_conv_to_zeros(layer)
            self.blocks.append(layer)
        if self.num_skip_connection > 1:
            layer = nn.Conv2d(self.foot.hidden_dim_8x, self.output_size, 1)
            init_conv_to_zeros(layer)
            self.blocks.append(layer)

    def build_head(self, output_size, **kwargs):
        # 8s => k=16, s=8
        # 16s => k=32, s=16
        # 32s => k=64, s=32
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

        mapping = [
            (64, 32),
            (32, 16),
            (16, 8),
        ]

        k, s = mapping[self.num_skip_connection]
        layer = nn.ConvTranspose2d(
            output_size,
            output_size,
            k,
            stride=s,
            bias=False,
        )
        init_deconv_with_bilinear_kernel(layer)
        layer.requires_grad_(False)
        self.head.insert(0, layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8s => 3, 4, 6
        # 16s => 4, 6
        # 32s => 6
        features: list[torch.Tensor] = self.foot(x)

        y = self.blocks[0](features.pop())
        for h, block in zip(self.head[1:], self.blocks[1:]):
            # upsampling low-level features
            y = h(y)

            # feature fusion
            y += CenterCrop(y.shape[-2:])(block(features.pop()))

        y = self.head[0](y)
        return CenterCrop(x.shape[-2:])(y)
