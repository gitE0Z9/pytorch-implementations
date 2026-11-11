from collections import namedtuple
from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import ResBlock
from torchlake.common.models.model_base import ModelBase

from .network import BottleNeck, BottleNeckB, ConvBlock, ResBlockD

ConvBlockArgs = namedtuple(
    "ConvBlockArgs",
    ("input_dim, output_dim, number_block, layer_cls"),
)

BottleneckArgs = namedtuple(
    "BottleneckArgs",
    ("input_dim, hidden_dim, expansion, number_block, layer_cls"),
)


class ResNet(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        pre_activation: bool = False,
        key: Literal[18, 34, 50, 101, 152] = 50,
        config: tuple[ConvBlockArgs | BottleneckArgs] | None = None,
    ):
        """ResNet

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            key (Literal[18, 34, 50, 101, 152], optional): key of configs. Defaults to 50.
            config (tuple[ConvBlockArgs | BottleneckArgs], optional): stage conifg, format is (input_dim, output_dim, number_block, layer_cls) or (input_dim, hidden_dim, expansion, number_block, layer_cls). Defaults to None.
        """
        self.pre_activation = pre_activation
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self):
        if len(self.config[0]) == 4:
            return self.config[-1][1]
        else:
            return self.config[-1][1] * self.config[-1][2]

    @property
    def config(self) -> tuple[ConvBlockArgs | BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[ConvBlockArgs | BottleneckArgs]: parameters of layers
        """
        return (
            self._config
            or {
                18: (
                    ConvBlockArgs(64, 64, 2, ConvBlock),  # less block
                    ConvBlockArgs(64, 128, 2, ConvBlock),  # less block
                    ConvBlockArgs(128, 256, 2, ConvBlock),  # less block
                    ConvBlockArgs(256, 512, 2, ConvBlock),  # less block
                ),
                34: (
                    ConvBlockArgs(64, 64, 3, ConvBlock),  # narrower
                    ConvBlockArgs(64, 128, 4, ConvBlock),  # narrower
                    ConvBlockArgs(128, 256, 6, ConvBlock),  # narrower
                    ConvBlockArgs(256, 512, 3, ConvBlock),  # narrower
                ),
                #
                50: (
                    BottleneckArgs(64, 64, 4, 3, BottleNeck),
                    BottleneckArgs(256, 128, 4, 4, BottleNeck),
                    BottleneckArgs(512, 256, 4, 6, BottleNeck),
                    BottleneckArgs(1024, 512, 4, 3, BottleNeck),
                ),
                101: (
                    BottleneckArgs(64, 64, 4, 3, BottleNeck),
                    BottleneckArgs(256, 128, 4, 4, BottleNeck),
                    BottleneckArgs(512, 256, 4, 23, BottleNeck),  # more block
                    BottleneckArgs(1024, 512, 4, 3, BottleNeck),
                ),
                152: (
                    BottleneckArgs(64, 64, 4, 3, BottleNeck),
                    BottleneckArgs(256, 128, 4, 8, BottleNeck),  # more block
                    BottleneckArgs(512, 256, 4, 36, BottleNeck),  # more block
                    BottleneckArgs(1024, 512, 4, 3, BottleNeck),
                ),
            }[self.key]
        )

    def build_foot(self, input_channel: int):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.config[0].input_dim,
                7,
                stride=2,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        if self.pre_activation:
            self.foot.insert(2, nn.ReLU(True))

    def build_blocks(self):
        self.blocks = nn.Sequential()

        for stage_cfg in self.config:
            is_conv_block = isinstance(stage_cfg, ConvBlockArgs)
            if is_conv_block:
                in_c, out_c, num_block, layer_cls = stage_cfg
            else:
                in_c, hidden_dim, expansion_factor, num_block, layer_cls = stage_cfg
                out_c = hidden_dim * expansion_factor

            stage = nn.Sequential()
            for block_idx in range(num_block):
                stride = 1
                if block_idx == 0:
                    stride = 2
                else:
                    in_c = out_c

                if is_conv_block:
                    layer = layer_cls(
                        in_c,
                        out_c,
                        stride=stride,
                        pre_activation=self.pre_activation,
                    )
                else:
                    layer = layer_cls(
                        in_c,
                        hidden_dim,
                        stride=stride,
                        expansion_factor=expansion_factor,
                        pre_activation=self.pre_activation,
                    )
                stage.append(
                    ResBlock(
                        in_c,
                        out_c,
                        stride=stride,
                        block=layer,
                        activation=None if self.pre_activation else nn.ReLU(True),
                    )
                )

            self.blocks.append(stage)

        if self.pre_activation:
            self.blocks[-1].activation = nn.ReLU(True)


class ResNetB(ResNet):
    @property
    def config(self) -> tuple[BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[BottleneckArgs]: parameters of layers
        """
        return {
            50: (
                BottleneckArgs(64, 64, 4, 3, BottleNeckB),
                BottleneckArgs(256, 128, 4, 4, BottleNeckB),
                BottleneckArgs(512, 256, 4, 6, BottleNeckB),
                BottleneckArgs(1024, 512, 4, 3, BottleNeckB),
            ),
            101: (
                BottleneckArgs(64, 64, 4, 3, BottleNeckB),
                BottleneckArgs(256, 128, 4, 4, BottleNeckB),
                BottleneckArgs(512, 256, 4, 23, BottleNeckB),  # more block
                BottleneckArgs(1024, 512, 4, 3, BottleNeckB),
            ),
            152: (
                BottleneckArgs(64, 64, 4, 3, BottleNeckB),
                BottleneckArgs(256, 128, 4, 8, BottleNeckB),  # more block
                BottleneckArgs(512, 256, 4, 36, BottleNeckB),  # more block
                BottleneckArgs(1024, 512, 4, 3, BottleNeckB),
            ),
        }[self.key]


class ResNetC(ResNet):
    def build_foot(self, input_channel: int):
        first_hidden_dim = self.config[0].input_dim
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, first_hidden_dim, 3, stride=2),
            Conv2dNormActivation(first_hidden_dim, first_hidden_dim, 3),
            Conv2dNormActivation(first_hidden_dim, first_hidden_dim, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )


class ResNetD(ResNet):
    @property
    def config(self) -> tuple[BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[BottleneckArgs]: parameters of layers
        """
        return {
            50: (
                BottleneckArgs(64, 64, 4, 3, BottleNeckB),
                BottleneckArgs(256, 128, 4, 4, BottleNeckB),
                BottleneckArgs(512, 256, 4, 6, BottleNeckB),
                BottleneckArgs(1024, 512, 4, 3, BottleNeckB),
            ),
            101: (
                BottleneckArgs(64, 64, 4, 3, BottleNeckB),
                BottleneckArgs(256, 128, 4, 4, BottleNeckB),
                BottleneckArgs(512, 256, 4, 23, BottleNeckB),  # more block
                BottleneckArgs(1024, 512, 4, 3, BottleNeckB),
            ),
            152: (
                BottleneckArgs(64, 64, 4, 3, BottleNeckB),
                BottleneckArgs(256, 128, 4, 8, BottleNeckB),  # more block
                BottleneckArgs(512, 256, 4, 36, BottleNeckB),  # more block
                BottleneckArgs(1024, 512, 4, 3, BottleNeckB),
            ),
        }[self.key]

    def build_blocks(self):
        self.blocks = nn.Sequential()

        for (
            in_c,
            hidden_dim,
            expansion_factor,
            num_block,
            layer_cls,
        ) in self.config:
            block = nn.Sequential()
            for block_idx in range(num_block):
                out_c = hidden_dim * expansion_factor
                stride = 1
                if block_idx == 0:
                    stride = 2
                else:
                    in_c = out_c

                block.append(
                    ResBlockD(
                        in_c,
                        out_c,
                        stride=stride,
                        block=layer_cls(
                            in_c,
                            hidden_dim,
                            stride=stride,
                            expansion_factor=expansion_factor,
                            pre_activation=self.pre_activation,
                        ),
                        activation=None if self.pre_activation else nn.ReLU(True),
                    )
                )

            self.blocks.append(block)
