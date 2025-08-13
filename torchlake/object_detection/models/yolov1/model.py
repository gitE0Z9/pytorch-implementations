from math import prod
import torch
import torch.nn as nn
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.flatten import FlattenFeature

from ...constants.schema import DetectorContext


class YOLOV1(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        context: DetectorContext,
        dropout_prob: float = 0.5,
    ):
        """mimic original design of YOLOV1 in paper [1506.02640]

        Args:
            backbone (ExtractorBase): feature extractor
            context (DetectorContext): detector context
            dropout_prob (float, optional): dropout prob. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.context = context
        super().__init__(
            3,
            context.num_anchors * 5 + context.num_classes,
            foot_kwargs={
                "backbone": backbone,
            },
            head_kwargs={
                "output_shape": (7, 7),
            },
        )
        # self.init_weight()

    @property
    def feature_dim(self) -> int:
        return 256 * 7 * 7

    # def init_weight(self):
    #     for layer in self.neck.children():
    #         if isinstance(layer, Conv2dNormActivation):
    #             torch.nn.init.kaiming_normal_(layer[0].weight)

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self):
        # like paper, add 4 convs after backbone
        self.neck = nn.Sequential(
            Conv2dNormActivation(
                self.foot.feature_dims[-1],
                1024,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            Conv2dNormActivation(
                1024,
                1024,
                3,
                stride=2,
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
            Conv2dNormActivation(
                1024,
                1024,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            Conv2dNormActivation(
                1024,
                256,
                3,
                norm_layer=None,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.Dropout(self.dropout_prob),
        )

    def build_head(self, output_size: int, **kwargs):
        output_shape = kwargs.pop("output_shape")

        self.head = nn.Sequential(
            FlattenFeature(reduction=None),
            nn.Linear(self.feature_dim, output_size * prod(output_shape)),
            nn.Unflatten(-1, (output_size, *output_shape)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x).pop()
        y = self.neck(y)

        return self.head(y)


class YOLOV1Modified(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        context: DetectorContext,
        dropout_prob: float = 0.5,
    ):
        """Accept any backbone and lighter convolution head

        Args:
            backbone (ExtractorBase): _description_
            context (DetectorContext): _description_
            dropout_prob (float, optional): _description_. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.context = context
        super().__init__(
            3,
            context.num_anchors * 5 + context.num_classes,
            foot_kwargs={
                "backbone": backbone,
            },
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self):
        # like paper, add 4 convs after backbone
        self.neck = nn.Sequential(
            Conv2dNormActivation(
                self.foot.feature_dims[-1],
                1024,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            Conv2dNormActivation(
                1024,
                1024,
                3,
                stride=2,
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
            Conv2dNormActivation(
                1024,
                1024,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            Conv2dNormActivation(
                1024,
                256,
                3,
                norm_layer=None,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.Dropout(self.dropout_prob),
        )

    def build_head(self, output_size: int):
        self.head = nn.Conv2d(256, output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x).pop()
        y = self.neck(y)

        return self.head(y)
