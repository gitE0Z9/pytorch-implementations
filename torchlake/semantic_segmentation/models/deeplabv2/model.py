# vgg16: bi_w = 4, bi_xy_std = 65, bi_rgb_std = 3, pos_w = 2, pos_xy_std = 2.
# resnet101: bi_w = 4, bi_xy_std = 67, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 1.

from functools import partial
import torch
from annotated_types import T
from torch import nn
from torchlake.common.models import ResNetFeatureExtractor, VGGFeatureExtractor
from torchvision.transforms import CenterCrop

from .network import ASPP, ShallowASPP


class DeepLabV2(nn.Module):

    def __init__(
        self,
        backbone: VGGFeatureExtractor | ResNetFeatureExtractor,
        output_size: int = 1,
        dilations: list[int] = [6, 12, 18, 24],
        feature_dim: int = 512,
    ):
        """Fully Convolutional Networks for Semantic Segmentation in paper [1605.06211v1]

        Args:
            backbone (VGGFeatureExtractor | ResNetFeatureExtractor): either vgg family or resnet family.
            output_size (int, optional): output size. Defaults to 1.
            dilations (list[int], optional): dilation size of ASPP, for ASPP-S, it is [2,4,8,12], ASPP-L is default value. Defaults to [6, 12, 18, 24].
            feature_dim (int, optional): input channel of ASPP. Defaults to 512.
        """
        super(DeepLabV2, self).__init__()
        self.dilations = dilations
        self.feature_dim = feature_dim
        self.backbone = self.build_backbone(backbone)
        self.head = self.build_head(output_size)

    def build_backbone(
        self,
        backbone: VGGFeatureExtractor | ResNetFeatureExtractor,
    ) -> VGGFeatureExtractor | ResNetFeatureExtractor:
        """build backbone

        Args:
            backbone (VGGFeatureExtractor | ResNetFeatureExtractor): either vgg family or resnet family.

        Returns:
            VGGFeatureExtractor | ResNetFeatureExtractor: feature extractor
        """
        if isinstance(backbone, VGGFeatureExtractor):
            feature_extractor = backbone.feature_extractor
            # stage 5 convs
            for i in range(1, 4):
                conv_layer: nn.Conv2d = feature_extractor[-1 - i * 2]
                conv_layer.dilation = (2, 2)
                conv_layer.padding = (2, 2)

            # skip subsampling and keep 8x
            stage = 0
            for layer in feature_extractor:
                if isinstance(layer, nn.MaxPool2d):
                    stage += 1
                    layer.padding = (1, 1)
                    layer.kernel_size = (3, 3)

                    # stage 4, 5
                    if stage >= 4:
                        layer.stride = (1, 1)

            backbone.forward = partial(backbone.forward, target_layer_names=["5_1"])
            self.feature_dim = 512

        elif isinstance(backbone, ResNetFeatureExtractor):
            feature_extractor = backbone.feature_extractor
            for key, layer in feature_extractor[3].named_modules():
                layer: nn.Conv2d
                if "conv2" in key:
                    layer.dilation, layer.padding, layer.stride = (2, 2), (2, 2), (1, 1)
                elif "downsample.0" in key:
                    layer.stride = (1, 1)
            for key, layer in feature_extractor[4].named_modules():
                if "conv2" in key:
                    layer.dilation, layer.padding, layer.stride = (4, 4), (4, 4), (1, 1)
                elif "downsample.0" in key:
                    layer.stride = (1, 1)

            backbone.forward = partial(backbone.forward, target_layer_names=["4_1"])
            self.feature_dim = 2048

        return backbone

    def build_head(self, output_size: int = 1) -> ShallowASPP | ASPP:
        """deeplab v2 use ASPP as default head, for resnet shallow ASPP is used as [config](http://liangchiehchen.com/projects/DeepLabv2_resnet.html)

        Args:
            output_size (int, optional): output size. Defaults to 1.

        Returns:
            ShallowASPP | ASPP: ASPP module
        """
        if isinstance(self.backbone, ResNetFeatureExtractor):
            return ShallowASPP(self.feature_dim, output_size, dilations=self.dilations)

        return ASPP(512, 1024, output_size, dilations=self.dilations)

    def eval(self: T) -> T:
        """during eval mode, deeplab v2 will upsample prediction to full size

        Args:
            self (T): _description_

        Returns:
            T: _description_
        """
        result = super().eval()

        # upsample every prediction
        result.upsample = nn.Upsample(
            scale_factor=8,
            mode="bilinear",
            align_corners=True,
        )

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.backbone.forward(x)
        y = self.head(features.pop())

        is_eval = not self.training
        if is_eval:
            cropper = CenterCrop(x.shape[2:])
            y = self.upsample(y)
            y = cropper(y)

        return y
