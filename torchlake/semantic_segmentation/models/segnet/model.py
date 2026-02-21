from __future__ import annotations

import torch
from torch import nn

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor

from .network import DecoderBlock


class SegNet(ModelBase):
    def __init__(self, backbone: ExtractorBase, output_size: int = 1):
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _: int, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                DecoderBlock(self.foot.hidden_dim_32x, self.foot.hidden_dim_16x, 3),
                DecoderBlock(self.foot.hidden_dim_16x, self.foot.hidden_dim_8x, 3),
                DecoderBlock(self.foot.hidden_dim_8x, self.foot.hidden_dim_4x, 3),
                DecoderBlock(self.foot.hidden_dim_4x, self.foot.hidden_dim_2x, 2),
                DecoderBlock(self.foot.hidden_dim_2x, self.foot.hidden_dim_2x, 2),
            ]
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.foot.hidden_dim_2x, output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, pooling_indices = self.foot(x)

        y = features.pop()
        for neck in self.neck:
            y = neck(y, pooling_indices.pop())

        return self.head(y)


class BayesianSegNet(SegNet):
    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        dropout_prob: float = 0.5,
        num_iter: int = 6,
    ):
        self.dropout_prob = dropout_prob
        self.num_iter = num_iter
        super().__init__(backbone, output_size)

    def build_foot(self, *args, **kwargs):
        super().build_foot(*args, **kwargs)

        if isinstance(self.foot, VGGFeatureExtractor):
            stages = self.foot.get_stage()
            stage4 = stages[-2]
            self.foot.feature_extractor.insert(
                stage4[-1] + (4 if self.foot.enable_bn else 3),
                nn.Dropout(p=self.dropout_prob),
            )
            self.foot.feature_extractor.append(
                nn.Dropout(p=self.dropout_prob),
            )
        else:
            raise NotImplementedError

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                DecoderBlock(
                    self.foot.hidden_dim_32x,
                    self.foot.hidden_dim_16x,
                    3,
                    dropout_prob=self.dropout_prob,
                ),
                DecoderBlock(
                    self.foot.hidden_dim_16x,
                    self.foot.hidden_dim_8x,
                    3,
                    dropout_prob=self.dropout_prob,
                ),
                DecoderBlock(self.foot.hidden_dim_8x, self.foot.hidden_dim_4x, 3),
                DecoderBlock(self.foot.hidden_dim_4x, self.foot.hidden_dim_2x, 2),
                DecoderBlock(self.foot.hidden_dim_2x, self.foot.hidden_dim_2x, 2),
            ]
        )

    def eval(self: BayesianSegNet) -> BayesianSegNet:
        result = super().eval()

        for _, m in result.named_modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return result

    def forward(
        self,
        x: torch.Tensor,
        output_uncertainty: bool = False,
    ) -> torch.Tensor:
        if self.training:
            return super().forward(x)

        outputs = []
        for _ in range(self.num_iter):
            outputs.append(super().forward(x))
        outputs = torch.stack(outputs)

        if output_uncertainty:
            return outputs.mean(0), outputs.var(0)
        else:
            return outputs.mean(0)
