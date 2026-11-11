from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import DepthwiseSeparableConv2d
from torchlake.common.models.model_base import ModelBase


class MobileNetV1(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
        config: tuple[tuple[int]] | None = None,
    ):
        """MobileNet version 1 [1704.04861v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
            config (tuple[tuple[int]], optional): block conifg, format is (input_channel, output_channel, stride, num_layer). Defaults to None.
        """
        self.width_multiplier = width_multiplier
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self):
        return int(1024 * self.width_multiplier)

    @property
    def config(self) -> tuple[tuple[int]]:
        # input_channel, output_channel, stride, num_layer
        return self._config or (
            (32, 64, 1, 1),
            (64, 128, 2, 1),
            (128, 128, 1, 1),
            (128, 256, 2, 1),
            (256, 256, 1, 1),
            (256, 512, 2, 1),
            (512, 512, 1, 5),
            (512, 1024, 2, 1),
            (1024, 1024, 2, 1),
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                int(32 * self.width_multiplier),
                stride=2,
            ),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for in_c, out_c, stride, num_layer in self.config:
            for _ in range(num_layer):
                self.blocks.append(
                    DepthwiseSeparableConv2d(
                        int(in_c * self.width_multiplier),
                        int(out_c * self.width_multiplier),
                        stride=stride,
                    )
                )
