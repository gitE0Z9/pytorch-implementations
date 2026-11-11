from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..mobilenet.model import MobileNetV1
from .network import InvertedResidualBlock


class MobileNetV2(MobileNetV1):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
        dropout_prob: float = 0.5,
        config: tuple[tuple[int]] | None = None,
    ):
        """MobileNet version 2 [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            config (tuple[tuple[int]], optional): block conifg, format is (input_channel, output_channel, stride, expansion_factor, num_layer). Defaults to None.
        """
        self.width_multiplier = width_multiplier
        self.dropout_prob = dropout_prob
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self):
        return int(1280 * min(self.width_multiplier, 1))

    @property
    def config(self) -> tuple[tuple[int]]:
        # input_channel, output_channel, stride, expansion_factor, num_layer
        return self._config or (
            (32, 16, 1, 1, 1),
            (16, 24, 2, 6, 2),
            (24, 32, 2, 6, 3),
            (32, 64, 2, 6, 4),
            (64, 96, 1, 6, 3),
            (96, 160, 2, 6, 3),
            (160, 320, 1, 6, 1),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for in_c, out_c, stride, expansion_ratio, num_layer in self.config:
            for l in range(num_layer):
                self.blocks.append(
                    InvertedResidualBlock(
                        int((in_c if l == 0 else out_c) * self.width_multiplier),
                        int(out_c * self.width_multiplier),
                        stride=stride if l == 0 else 1,
                        expansion_ratio=expansion_ratio,
                    )
                )

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(1, nn.Dropout(self.dropout_prob))
        self.head.insert(
            0,
            Conv2dNormActivation(int(320 * self.width_multiplier), self.feature_dim, 1),
        )
