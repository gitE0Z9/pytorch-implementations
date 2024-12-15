from typing import Literal

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import MLADecoder, PUPDecoder


class SETR(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        decoder: Literal["PUP", "MLA"] = "PUP",
    ):
        """Segmentation transformer"""
        super().__init__(
            3,
            output_size,
            foot_kwargs={
                "backbone": backbone,
            },
            head_kwargs={
                "decoder": decoder,
            },
        )

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("backbone")

    def build_head(self, output_size, **kwargs):
        decoder = kwargs.pop("decoder")
        input_channel = self.foot.feature_dims[-1]

        if decoder == "PUP":
            self.head = PUPDecoder(input_channel, output_size)
        elif decoder == "MLA":
            self.head = MLADecoder(input_channel, output_size)
