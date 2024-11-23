import torch
from torch import nn
from torchlake.common.models import L2Norm, VGGFeatureExtractor
from torchvision.ops import Conv2dNormActivation


class Backbone(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()
        self.backbone: VGGFeatureExtractor = self.build_backbone(trainable)
        self.norm = L2Norm(512, scale=20.0)

    def build_backbone(self, trainable: bool = True):
        backbone = VGGFeatureExtractor("vgg16", "conv", trainable=trainable)  # [:30]
        feature_extractor: nn.Sequential = backbone.feature_extractor

        for module in [
            # conv6 (6_1)
            Conv2dNormActivation(
                512,
                1024,
                3,
                padding=6,
                dilation=6,
                norm_layer=None,
            ),
            # conv7 (6_2)
            Conv2dNormActivation(
                1024,
                1024,
                1,
                norm_layer=None,
            ),
            # conv8 (6_3)
            Conv2dNormActivation(1024, 256, 1, norm_layer=None),
            # conv8,s2 (6_4)
            Conv2dNormActivation(
                256,
                512,
                3,
                stride=2,
                padding=1,
                norm_layer=None,
            ),
            # conv9 (6_5)
            Conv2dNormActivation(512, 128, 1, norm_layer=None),
            # conv9,s2 (6_6)
            Conv2dNormActivation(
                128,
                256,
                3,
                stride=2,
                padding=1,
                norm_layer=None,
            ),
            # conv10 (6_7)
            Conv2dNormActivation(256, 128, 1, norm_layer=None),
            # conv10,down (6_8)
            Conv2dNormActivation(
                128,
                256,
                3,
                stride=1,
                padding=0,
                norm_layer=None,
            ),
            # conv11 (6_9)
            Conv2dNormActivation(256, 128, 1, norm_layer=None),
            # conv11,down (6_10)
            Conv2dNormActivation(
                128,
                256,
                3,
                stride=1,
                padding=0,
                norm_layer=None,
            ),
        ]:
            feature_extractor.append(module)

        feature_extractor[16].ceil_mode = True
        (
            feature_extractor[30].kernel_size,
            feature_extractor[30].stride,
            feature_extractor[30].padding,
        ) = (
            (3, 3),
            (1, 1),
            (1, 1),
        )

        return backbone

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # 4: 38, 5: 19, 6: 19, 19, 10, 5, 3, 1
        features: list[torch.Tensor] = self.backbone(
            x,
            ["4_3", "6_2", "6_4", "6_6", "6_8", "6_10"],
        )
        features[0] = self.norm(features[0].relu_())
        return features


class RegHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        num_priors: int,
        coord_dims: int = 4,
    ):
        """_summary_

        Args:
            input_channel (int): input channel
            num_classes (int): number of classes
            num_priors (int): number of prior boxes
            coord_dims (int, optional): coordinate dimensions. Defaults to 4.
        """
        # mark
        self.num_priors = num_priors
        self.coord_dims = coord_dims
        self.num_classes = num_classes

        super().__init__()
        self.block = nn.Conv2d(
            input_channel,
            num_priors * (coord_dims + num_classes),
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
