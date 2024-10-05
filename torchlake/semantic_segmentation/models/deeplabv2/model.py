# vgg16: bi_w = 4, bi_xy_std = 65, bi_rgb_std = 3, pos_w = 2, pos_xy_std = 2.
# resnet101: bi_w = 4, bi_xy_std = 67, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 1.

import torch
from annotated_types import T
from torch import nn
from torchlake.common.models import ResNetFeatureExtractor, VGGFeatureExtractor
from torchvision.transforms import CenterCrop

from .network import ASPP


class DeepLabV2(nn.Module):

    def __init__(
        self,
        backbone: VGGFeatureExtractor | ResNetFeatureExtractor,
        output_size: int = 1,
        dilations: list[int] = [6, 12, 18, 24],
    ):
        """Fully Convolutional Networks for Semantic Segmentation in paper [1605.06211v1]

        Args:
            backbone (VGGFeatureExtractor | ResNetFeatureExtractor): either vgg family or resnet family.
            output_size (int, optional): output size. Defaults to 1.
        """
        super(DeepLabV2, self).__init__()
        self.backbone = self.build_backbone(backbone)
        self.head = ASPP(512, 1024, output_size, dilations=dilations)

    def build_backbone(
        self,
        backbone: VGGFeatureExtractor | ResNetFeatureExtractor,
    ) -> VGGFeatureExtractor:
        feature_layers = backbone.feature_extractor
        # stage 5 convs
        for i in range(1, 4):
            conv_layer: nn.Conv2d = feature_layers[-1 - i * 2]
            conv_layer.dilation = (2, 2)
            conv_layer.padding = (2, 2)

        # skip subsampling and keep 8x
        stage = 0
        for layer in feature_layers:
            if isinstance(layer, nn.MaxPool2d):
                stage += 1
                layer.padding = (1, 1)
                layer.kernel_size = (3, 3)

                # stage 4, 5
                if stage >= 4:
                    layer.stride = (1, 1)

        return backbone

    def eval(self: T) -> T:
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
        features: list[torch.Tensor] = self.backbone(x, ["5_1"])
        y = self.head(features.pop())

        is_eval = not self.training
        if is_eval:
            cropper = CenterCrop(x.shape[2:])
            y = self.upsample(y)
            y = cropper(y)

        return y
