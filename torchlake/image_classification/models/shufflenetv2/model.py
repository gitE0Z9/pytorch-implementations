from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

from .network import ResBlock


class ShuffleNetV2(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        groups: int = 1,
        scale_factor: Literal[0.5, 1, 1.5, 2] = 1,
        config: tuple[tuple[int]] | None = None,
    ):
        """ShuffleNet version 2 [1807.11164v1]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            groups (int, optional): number of groups of channel shuffle. Defaults to 1.
            scale_factor (Literal[0.5, 1, 1.5, 2], optional): scale factor s. Defaults to 1.
            config (tuple[tuple[int]], optional): stage config, format is (input_channel, output_channel, num_layer). Defaults to None.
        """
        self.groups = groups
        # for identity check
        self.scale_factor = scale_factor
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 1024 if self.scale_factor < 2 else 2048

    @property
    def config(self) -> tuple[tuple[int]]:
        # input_channel, output_channel, num_layer
        return (
            self._config
            or {
                0.5: (
                    (24, 48, 4),
                    (48, 96, 8),
                    (96, 192, 4),
                ),
                1: (
                    (24, 116, 4),
                    (116, 232, 8),
                    (232, 464, 4),
                ),
                1.5: (
                    (24, 176, 4),
                    (176, 352, 8),
                    (352, 704, 4),
                ),
                2: (
                    (24, 244, 4),
                    (244, 488, 8),
                    (488, 976, 4),
                ),
            }[self.scale_factor]
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.config[0][0],
                3,
                stride=2,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for in_c, out_c, num_layer in self.config:
            stage = nn.Sequential()
            for layer_idx in range(num_layer):
                stage.append(
                    ResBlock(
                        in_c if layer_idx == 0 else out_c,
                        out_c,
                        stride=2 if layer_idx == 0 else 1,
                        groups=self.groups,
                    )
                )

            self.blocks.append(stage)

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(
            0,
            Conv2dNormActivation(self.config[-1][1], self.feature_dim, 1),
        )
