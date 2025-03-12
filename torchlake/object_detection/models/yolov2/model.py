import torch
import torch.nn as nn
from torchlake.common.models import StackedPatch2d
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from ...constants.schema import DetectorContext


class YOLOV2(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        context: DetectorContext,
        backbone_feature_dims: tuple[int, int],
    ):
        self.context = context
        self.passthrough_feature_dim, self.neck_feature_dim = backbone_feature_dims
        super().__init__(
            3,
            1,
            foot_kwargs={
                "backbone": backbone,
            },
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self):
        self.neck = nn.ModuleDict(
            {
                "passthrough": nn.Sequential(
                    Conv2dNormActivation(
                        self.passthrough_feature_dim,
                        self.passthrough_feature_dim // 8,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    # reorg layer
                    StackedPatch2d(2),
                ),
                "neck": nn.Sequential(
                    Conv2dNormActivation(
                        self.neck_feature_dim,
                        1024,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        1024,
                        1024,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                ),
            }
        )

    def build_head(self, _):
        self.head = nn.Sequential(
            Conv2dNormActivation(
                1024 + self.passthrough_feature_dim // 2,
                1024,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.Conv2d(
                1024,
                self.context.num_anchors * (self.context.num_classes + 5),
                1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv_5, conv_6
        skip, y = self.foot(x)

        # B, 64*4, 13, 13
        skip = self.neck["passthrough"](skip)

        # B, 1024, 13, 13
        y = self.neck["neck"](y)

        # 1280, 13, 13
        y = torch.cat([y, skip], dim=1)

        # B, A*(C+5), 13, 13
        y = self.head(y)

        return y
