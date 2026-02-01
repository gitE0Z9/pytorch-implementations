import torch

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import DUC


class HDCDUC(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int,
        output_stride: int = 8,
    ):
        self.output_stride = output_stride
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("4_1",))

    def build_head(self, output_size, **kwargs):
        self.head = DUC(
            self.foot.hidden_dim_32x,
            output_size,
            self.output_stride,
            (6, 12, 18, 24),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        return self.head(y.pop())
