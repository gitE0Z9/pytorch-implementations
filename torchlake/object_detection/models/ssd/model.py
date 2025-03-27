import torch
from torch import nn
from torchlake.common.models import L2Norm
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from ...constants.schema import DetectorContext
from .network import RegHead


class SSD(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        context: DetectorContext,
    ):
        self.context = context
        # 1 is background
        super().__init__(
            3,
            context.num_classes + 1,
            foot_kwargs={
                "backbone": backbone,
            },
        )

    def build_foot(self, _, **kwargs):
        backbone: ExtractorBase = kwargs.pop("backbone")
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

        backbone.fix_target_layers(["4_3", "6_2", "6_4", "6_6", "6_8", "6_10"])
        self.foot = backbone

    def build_neck(self):
        self.neck = L2Norm(512, scale=20.0)

    def build_head(self, output_size):
        self.head = nn.ModuleList(
            [
                RegHead(512, 4, output_size),
                RegHead(1024, 6, output_size),
                RegHead(512, 6, output_size),
                RegHead(256, 6, output_size),
                RegHead(256, 4, output_size),
                RegHead(256, 4, output_size),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        y = []

        # 4: 38, 5: 19, 6: 19, 19, 10, 5, 3, 1
        features: list[torch.Tensor] = self.foot(x)
        features[0] = self.neck(features[0])

        for feature, head in zip(features, self.head):
            head: RegHead
            # batch, num anchor?*grid?*grid?, 4 + 1 + num class
            y_i: torch.Tensor = (
                head(feature)
                .reshape(
                    batch_size,
                    head.num_priors,
                    head.coord_dims + head.num_classes,
                    -1,
                )
                .permute(0, 1, 3, 2)
                .reshape(batch_size, -1, head.coord_dims + head.num_classes)
            )

            y.append(y_i)

        # batch, num anchor*grid*grid, 4 + 1 + num class
        return torch.cat(y, 1)
