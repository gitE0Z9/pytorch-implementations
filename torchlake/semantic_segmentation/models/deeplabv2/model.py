# vgg16: bi_w = 4, bi_xy_std = 65, bi_rgb_std = 3, pos_w = 2, pos_xy_std = 2.
# resnet101: bi_w = 4, bi_xy_std = 67, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 1.

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor
from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor

from .network import ASPP, ShallowASPP


class DeepLabV2(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        dilations: Sequence[int] = (6, 12, 18, 24),
        enable_shallow_aspp: bool = False,
    ):
        """DeepLab v2 in paper [1606.00915v2]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            dilations (Sequence[int], optional): dilation size of ASPP, for ASPP-S, it is [2,4,8,12], ASPP-L is default value. Defaults to [6, 12, 18, 24].
        """
        self.dilations = dilations
        self.enable_shallow_aspp = enable_shallow_aspp
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

        if isinstance(self.foot, VGGFeatureExtractor):
            self.foot.fix_target_layers(("5_1",))
        elif isinstance(self.foot, ResNetFeatureExtractor):
            self.foot.fix_target_layers(("4_1",))

    def build_head(self, output_size, **kwargs):
        """
        deeplab v2 use ASPP as default head

        for resnet shallow ASPP is used as [config](http://liangchiehchen.com/projects/DeepLabv2_resnet.html)

        Args:
            output_size (int, optional): output size. Defaults to 1.
        """
        if self.enable_shallow_aspp:
            self.head = nn.Sequential(
                ShallowASPP(
                    self.foot.feature_dim,
                    output_size,
                    dilations=self.dilations,
                )
            )
            return

        # shallow head perform worse, yet heavy head is too heavy about 122M parameters
        if isinstance(self.foot, VGGFeatureExtractor):
            layer = ASPP(
                self.foot.hidden_dim_32x,
                1024,
                output_size,
                dilations=self.dilations,
            )

            if self.foot.enable_fc1 and self.foot.enable_fc2:
                for block in layer.blocks:
                    block[0][0].weight.data.copy_(
                        self.foot.feature_extractor[-7].weight.data
                    )
                    block[0][0].bias.data.copy_(
                        self.foot.feature_extractor[-7].bias.data
                    )
                    block[2][0].weight.data.copy_(
                        self.foot.feature_extractor[-4].weight.data
                    )
                    block[2][0].bias.data.copy_(
                        self.foot.feature_extractor[-4].bias.data
                    )

                self.foot.feature_extractor = self.foot.feature_extractor[:-7]

            self.head = nn.Sequential(layer)
        else:
            self.head = nn.Sequential(
                ASPP(
                    self.foot.feature_dim,
                    1024,
                    output_size,
                    dilations=self.dilations,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.foot(x)
        y = self.head(features.pop())

        if not self.training:
            cropper = CenterCrop(x.shape[2:])
            y = F.interpolate(
                y,
                mode="bilinear",
                scale_factor=8,
                align_corners=True,
                # size=x.shape[2:],
            )
            y = cropper(y)

        return y
