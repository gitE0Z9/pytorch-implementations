from functools import partial

import torch
import torch.nn as nn
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from ...constants.schema import DetectorContext


class YOLOV1Original(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        context: DetectorContext,
        dropout_prob: float = 0.5,
    ):
        """mimic original design of YOLOV1 in paper [1506.02640]

        Args:
            backbone (ExtractorBase): _description_
            context (DetectorContext): _description_
            dropout_prob (float, optional): _description_. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.context = context
        self.output_size = context.num_anchors * 5 + context.num_classes
        super().__init__(
            3,
            7 * 7 * self.output_size,
            foot_kwargs={
                "backbone": backbone,
            },
        )
        # self.init_weight()

    @property
    def feature_dim(self) -> int:
        return 256

    # def init_weight(self):
    #     for layer in self.neck.children():
    #         if isinstance(layer, Conv2dNormActivation):
    #             torch.nn.init.kaiming_normal_(layer[0].weight)

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.forward = partial(
            self.foot.forward,
            target_layer_names=["4_1"],
        )

    def build_neck(self):
        # like paper, add 4 convs after backbone
        self.neck = nn.Sequential(
            Conv2dNormActivation(
                512,  # TODO: convert back to 1024 for extraction
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
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.Dropout(self.dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x).pop()
        y = self.neck(y)

        return self.head(y).view(-1, self.output_size, 7, 7)


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
        self.output_size = context.num_anchors * 5 + context.num_classes
        super().__init__(
            3,
            self.output_size,
            foot_kwargs={
                "backbone": backbone,
            },
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.forward = partial(
            self.foot.forward,
            target_layer_names=["4_1"],
        )

    def build_neck(self):
        # like paper, add 4 convs after backbone
        self.neck = nn.Sequential(
            Conv2dNormActivation(
                512,
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
            nn.Dropout(self.dropout_prob),
        )

    def build_head(self, _):
        self.head = nn.Conv2d(1024, self.output_size, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x).pop()
        y = self.neck(y)

        return self.head(y)
