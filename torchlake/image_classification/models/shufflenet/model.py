from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase
from torchlake.common.utils.numerical import make_divisible

from .network import ResBlock


class ShuffleNet(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        groups: Literal[1, 2, 3, 4, 8] = 1,
        scale_factor: float = 1,
        config: tuple[tuple[int]] | None = None,
    ):
        """ShuffleNet [1707.01083v2]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            groups (Literal[1, 2, 3, 4, 8], optional): number of groups of channels. Defaults to 1.
            scale_factor (float, optional): scale factor s. Defaults to 1.
            config (tuple[tuple[int]], optional): stage config, format is (input_channel, output_channel, num_layer). Defaults to None.
        """
        self.groups = groups
        # for identity check
        self.scale_factor = scale_factor
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return self.stage_channels[-1]

    @property
    def stem_channel(self) -> int:
        """scaled stem channel, divisible by groups"""
        return make_divisible(self.scale_factor * self.config[0][0], self.groups)

    @property
    def stage_channels(self) -> tuple[int]:
        """scaled output channel of each stage,
        divisible by 4 x groups so that the bottleneck (1/4 of output) splits evenly into groups
        """
        return tuple(
            make_divisible(self.scale_factor * out_c, 4 * self.groups)
            for _, out_c, _ in self.config
        )

    @property
    def config(self) -> tuple[tuple[int]]:
        # input_channel, output_channel, num_layer
        return (
            self._config
            or {
                1: (
                    (24, 144, 4),
                    (144, 288, 8),
                    (288, 576, 4),
                ),
                2: (
                    (24, 200, 4),
                    (200, 400, 8),
                    (400, 800, 4),
                ),
                3: (
                    (24, 240, 4),
                    (240, 480, 8),
                    (480, 960, 4),
                ),
                4: (
                    (24, 272, 4),
                    (272, 544, 8),
                    (544, 1088, 4),
                ),
                8: (
                    (24, 384, 4),
                    (384, 768, 8),
                    (768, 1536, 4),
                ),
            }[self.groups]
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.stem_channel,
                3,
                stride=2,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        in_c = self.stem_channel
        for stage_idx, ((_, _, num_layer), out_c) in enumerate(
            zip(self.config, self.stage_channels)
        ):
            stage = nn.Sequential()
            for layer_idx in range(num_layer):
                stage.append(
                    ResBlock(
                        in_c if layer_idx == 0 else out_c,
                        out_c,
                        stride=2 if layer_idx == 0 else 1,
                        groups=self.groups,
                        # paper: no group conv on the first pointwise layer of stage 2
                        group_first_pointwise=not (stage_idx == 0 and layer_idx == 0),
                    )
                )

            self.blocks.append(stage)
            in_c = out_c
