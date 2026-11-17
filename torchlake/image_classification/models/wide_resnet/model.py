from typing import Literal

from torch import nn
from torchlake.common.models import ResBlock

from ..resnet.model import ResNet, ConvBlockArgs, BottleneckArgs
from ..resnet.network import BottleNeck
from .network import DropoutConvBlock


class WideResNet(ResNet):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        deepening_factor: int = 1,
        widening_factor: int = 1,
        dropout_prob: float = 0,
        pre_activation: bool = True,
        key: Literal["wrn", 50, 101, 152] = "wrn",
        config: tuple[ConvBlockArgs | BottleneckArgs] | None = None,
    ):
        """Wide ResNet in paper [1605.07146v4]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to True.
            deepening_factor (int, optional): depth muliplier l in paper [1605.07146v4], number of layer = 6 * k + 4. Defaults to 1.
            widening_factor (int, optional): width muliplier k in paper [1605.07146v4]. Defaults to 1.
            dropout_prob (float, optional): dropout probability of convolution block. Defaults to 0.
            key (Literal["wrn", 50, 101, 152], optional): key of configs. Defaults to "wrn".
            config (tuple[ConvBlockArgs | BottleneckArgs], optional): stage conifg, format is (input_dim, output_dim, number_block, layer_cls) or (input_dim, hidden_dim, expansion, number_block, layer_cls). Defaults to None.
        """
        self.dropout_prob = dropout_prob
        self.deepening_factor = deepening_factor
        self.widening_factor = widening_factor
        super().__init__(
            input_channel,
            output_size,
            pre_activation=pre_activation,
            key=key,
            config=config,
        )

    @property
    def feature_dim(self):
        last_cfg: ConvBlockArgs | BottleneckArgs = self.config[-1]
        return int(last_cfg.output_dim * self.widening_factor)

    @property
    def config(self) -> tuple[tuple[int]]:
        """parameter of layers

        Returns:
            tuple[tuple[int]]: parameters of layers
        """
        return (
            self._config
            or {
                "wrn": (
                    ConvBlockArgs(16, 16, 1, DropoutConvBlock),
                    ConvBlockArgs(16, 32, 1, DropoutConvBlock),
                    ConvBlockArgs(32, 64, 1, DropoutConvBlock),
                ),
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
        stem_cfg: ConvBlockArgs | BottleneckArgs = self.config[0]
        self.foot = nn.Sequential(
            DropoutConvBlock(
                input_channel,
                stem_cfg.output_dim,
                pre_activation=self.pre_activation,
            ),
        )

    def build_blocks(self):
        self.blocks = nn.Sequential()

        for stage_idx, stage_cfg in enumerate(self.config):
            is_conv_block = isinstance(stage_cfg, ConvBlockArgs)
            if is_conv_block:
                in_c, out_c, num_block, layer_cls = stage_cfg
            else:
                in_c, hidden_dim, expansion_factor, num_block, layer_cls = stage_cfg
                out_c = hidden_dim * expansion_factor

            stage = nn.Sequential()
            if stage_idx != 0:
                in_c *= self.widening_factor
            out_c *= self.widening_factor

            for block_idx in range(num_block * self.deepening_factor):
                stride = 1
                if block_idx == 0:
                    stride = 2 if stage_idx > 0 else 1
                else:
                    in_c = out_c

                if is_conv_block:
                    layer = layer_cls(
                        in_c,
                        out_c,
                        stride=stride,
                        pre_activation=self.pre_activation,
                        dropout_prob=self.dropout_prob,
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
                        block=layer,
                        stride=stride,
                        activation=None if self.pre_activation else nn.ReLU(True),
                    )
                )

            self.blocks.append(stage)
