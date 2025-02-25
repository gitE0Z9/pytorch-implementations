from typing import Literal

import torch
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase


class NeuralStyleTransfer(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        content_layer_names: list[str],
        style_layer_names: list[str],
    ):
        super().__init__(None, None, foot_kwargs={"backbone": backbone})
        self.content_layer_names = content_layer_names
        self.style_layer_names = style_layer_names

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("backbone")

    def build_head(self, _, **kwargs):
        pass

    def forward(
        self,
        img: torch.Tensor,
        type: Literal["content", "style"],
    ) -> list[torch.Tensor]:
        if type == "content":
            return self.foot(img, self.content_layer_names)
        elif type == "style":
            return self.foot(img, self.style_layer_names)
        else:
            raise NotImplementedError
