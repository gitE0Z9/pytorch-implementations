from typing import Literal

from torch import nn
from torchlake.common.models import ResBlock

from ..resnet.model import ResNet
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
        config: tuple[tuple[int]] | None = None,
    ):
        """Wide ResNet in paper [1605.07146v4]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to True.
            deepening_factor (int, optional): depth muliplier l in paper [1605.07146v4], number of layer = 6 * k + 4. Defaults to 1.
            widening_factor (int, optional): width muliplier k in paper [1605.07146v4]. Defaults to 1.
            dropout_prob (float, optional): dropout probability of convolution block. Defaults to 0.
            config (tuple[tuple[int]], optional): stage conifg, format is (input_dim, output_dim, num_block). Defaults to None.
        """
        self.dropout_prob = dropout_prob
        self.deepening_factor = deepening_factor
        self.widening_factor = widening_factor
        self._config = config

        super().__init__(input_channel, output_size, pre_activation)

    @property
    def config(self) -> tuple[tuple[int]]:
        """parameter of layers

        Returns:
            tuple[tuple[int | nn.Module]]: parameters of layers
        """
        # input_dim, output_dim, num_block
        return self._config or (
            (16, 16, 1),
            (16, 32, 1),
            (32, 64, 1),
        )

    def build_foot(self, input_channel):
        self.foot = nn.Sequential(
            DropoutConvBlock(
                input_channel,
                16,
                pre_activation=self.pre_activation,
            ),
        )

    def build_blocks(self):
        self.blocks = nn.Sequential()

        for stage_idx, (in_c, out_c, num_block) in enumerate(self.config):
            block = nn.Sequential()

            if stage_idx != 0:
                in_c *= self.widening_factor
            out_c *= self.widening_factor
            num_block *= self.deepening_factor

            for layer_idx in range(num_block):
                stride = 1
                if layer_idx == 0:
                    stride = 2
                else:
                    in_c = out_c

                block.append(
                    ResBlock(
                        in_c,
                        out_c,
                        DropoutConvBlock(
                            in_c,
                            out_c,
                            stride=stride,
                            pre_activation=self.pre_activation,
                            dropout_prob=self.dropout_prob,
                        ),
                        stride=stride,
                    )
                )

            self.blocks.append(block)


class WideResNetWithBottleneck(ResNet):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        widening_factor: int = 1,
        pre_activation: bool = True,
        key: Literal[50, 101, 152] = 50,
        config: tuple[tuple[int]] | None = None,
    ):
        """Bottleneck version of Wide ResNet in paper [1605.07146v4]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to True.
            widening_factor (int, optional): width muliplier k in paper [1605.07146v4]. Defaults to 1.
            key (Literal[50, 101, 152], optional): key of configs. Defaults to 50.
            config (tuple[tuple[int]], optional): stage conifg, format is (input_dim, hidden_dim, expansion_factor, number_block). Defaults to None.
        """
        self.widening_factor = widening_factor

        super().__init__(
            input_channel,
            output_size,
            pre_activation=pre_activation,
            key=key,
            config=config,
        )

    @property
    def config(self) -> tuple[tuple[int]]:
        # input_dim, hidden_dim, expansion_factor, number_block
        return self._config or {
            50: (
                (64, 64, 4, 3),
                (256, 128, 4, 4),
                (512, 256, 4, 6),
                (1024, 512, 4, 3),
            ),
            101: (
                (64, 64, 4, 3),
                (256, 128, 4, 4),
                (512, 256, 4, 23),  # more block
                (1024, 512, 4, 3),
            ),
            152: (
                (64, 64, 4, 3),
                (256, 128, 4, 8),  # more block
                (512, 256, 4, 36),  # more block
                (1024, 512, 4, 3),
            ),
        }

    def build_blocks(self):
        self.blocks = nn.Sequential()

        for stage_idx, (
            in_c,
            hidden_dim,
            expansion_factor,
            num_block,
        ) in enumerate(self.config):
            block = nn.Sequential()

            if stage_idx != 0:
                in_c *= self.widening_factor
            hidden_dim *= self.widening_factor
            out_c = hidden_dim * expansion_factor

            for layer_idx in range(num_block):
                stride = 1
                if layer_idx == 0:
                    stride = 2
                else:
                    in_c = out_c

                block.append(
                    ResBlock(
                        in_c,
                        out_c,
                        BottleNeck(
                            in_c,
                            hidden_dim,
                            stride=stride,
                            expansion_factor=expansion_factor,
                            pre_activation=self.pre_activation,
                        ),
                        stride=stride,
                    )
                )

            self.blocks.append(block)
