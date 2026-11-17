from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

from .network import DenseBlock, TransitionBlock


class DenseNet(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        growth_rate: int = 32,
        compression_ratio: float = 0.5,
        key: Literal[121, 169, 201, 264] = 169,
        config: tuple[int] | None = None,
    ):
        """DenseNet [1608.06993v5]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            growth_rate (int, optional): growth rate k. Defaults to 32.
            compression_ratio (float, optional): compression ratio of transition block. Defaults to 0.5.
            key (Literal[121, 169, 201, 264], optional): key of configs. Defaults to 169.
            config (tuple[int], optional): stage config, . Defaults to None.
        """
        self.growth_rate = growth_rate
        self.compression_ratio = compression_ratio
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        # output of the last dense block, available after build_blocks
        return self.blocks[-1][0].output_channel

    @property
    def config(self) -> tuple[int]:
        # num_layer
        return (
            self._config
            or {
                121: (6, 12, 24, 16),
                169: (6, 12, 32, 32),
                201: (6, 12, 48, 32),
                264: (6, 12, 64, 48),
            }[self.key]
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 7, stride=2),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()

        input_channel = 64
        for block_idx, num_layer in enumerate(self.config):
            dense_block = DenseBlock(
                input_channel,
                num_layer,
                growth_rate=self.growth_rate,
            )
            stage = nn.Sequential(dense_block)

            if block_idx != len(self.config) - 1:
                output_channel = dense_block.output_channel
                compressed_channel = int(output_channel * self.compression_ratio)
                stage.append(TransitionBlock(output_channel, compressed_channel))
                input_channel = compressed_channel

            self.blocks.append(stage)

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        # dense layers are pre-activated, normalize the last output
        self.head.insert(0, nn.BatchNorm2d(self.feature_dim))
