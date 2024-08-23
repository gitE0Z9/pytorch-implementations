from copy import deepcopy
from torch import nn
from torchlake.common.models import ResBlock

from ..resnet.model import ResNet
from .network import BottleNeck, DropoutConvBlock

# input, output, base?, num_repeat, block_type
CONFIGS = [
    [16, 16, 16, 1, DropoutConvBlock],
    [16, 32, 32, 1, DropoutConvBlock],
    [32, 64, 64, 1, DropoutConvBlock],
]


class WideResNet(ResNet):
    configs = CONFIGS

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        pre_activation: bool = True,
        deepening_factor: int = 1,
        widening_factor: int = 1,
        enable_dropout: bool = False,
        dropout_prob: float = 0.5,
    ):
        """Wide ResNet in paper [1605.07146v4]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, Defaults True): put activation before convolution layer in paper[1603.05027v3]
            deepening_factor (int, Defaults 1): depth muliplier l in paper [1605.07146v4], number of layer = 6 * k + 4. Defaults to 1.
            widening_factor (int, Defaults 1): width muliplier k in paper [1605.07146v4]. Defaults to 1.
            enable_dropout (bool, Defaults False): enable dropout layer between layers. Defaults to False.
            dropout_prob (float, Defaults 0.5): dropout probability of convolution block. Defaults to 0.5.
        """
        self.enable_dropout = enable_dropout
        self.dropout_prob = dropout_prob if self.enable_dropout else 0
        self.deepening_factor = deepening_factor
        self.widening_factor = widening_factor

        # modify config
        for stage_index, param in enumerate(self.config):
            in_c, out_c, base_c, num_repeat, block_class = param

            if stage_index != 0:
                in_c *= widening_factor

            out_c *= widening_factor
            base_c *= widening_factor
            num_repeat *= deepening_factor

            self.config[stage_index] = [in_c, out_c, base_c, num_repeat, block_class]

        super(WideResNet, self).__init__(
            input_channel,
            output_size,
            pre_activation,
        )

        # XXX: bad practice
        delattr(self, "key")

    @property
    def config(self) -> list[list[int | nn.Module]]:
        """parameter of layers

        Returns:
            list[list[int | nn.Module]]: parameters of layers
        """
        return deepcopy(self.configs)

    def build_blocks(self):
        """build blocks"""
        for block_index, (
            input_channel,
            output_channel,
            block_base_channel,
            num_repeat,
            block_class,
        ) in enumerate(self.config):
            layers = nn.Sequential()
            for layer_index in range(num_repeat):
                layers.append(
                    ResBlock(
                        input_channel if layer_index == 0 else output_channel,
                        output_channel,
                        block_class(
                            input_channel=(
                                input_channel if layer_index == 0 else output_channel
                            ),
                            block_base_channel=block_base_channel,
                            stride=2 if layer_index == 0 else 1,
                            pre_activation=self.pre_activation,
                            dropout_prob=self.dropout_prob,
                        ),
                        stride=2 if layer_index == 0 else 1,
                    )
                )

            setattr(self, f"block{block_index+1}", layers)


# input, output, base?, number_block, block_type
BOTTLENECK_CONFIGS = {
    50: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 4, BottleNeck],
        [512, 1024, 256, 6, BottleNeck],
        [1024, 2048, 512, 3, BottleNeck],
    ],
    101: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 4, BottleNeck],
        [512, 1024, 256, 23, BottleNeck],  # more block
        [1024, 2048, 512, 3, BottleNeck],
    ],
    152: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 8, BottleNeck],  # more block
        [512, 1024, 256, 36, BottleNeck],  # more block
        [1024, 2048, 512, 3, BottleNeck],
    ],
}


class BottleneckWideResNet(ResNet):
    configs = BOTTLENECK_CONFIGS

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        pre_activation: bool = True,
        key: int = 50,
        widening_factor: int = 1,
    ):
        """Bottleneck version of Wide ResNet in paper [1605.07146v4]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, Defaults True): put activation before convolution layer in paper[1603.05027v3]
            key (int, optional): key of configs. Defaults to 50.
            widening_factor (int, Defaults 1): width muliplier k in paper [1605.07146v4]. Defaults to 1.
        """
        self.widening_factor = widening_factor

        super(BottleneckWideResNet, self).__init__(
            input_channel,
            output_size,
            pre_activation,
            key,
        )

    def build_blocks(self):
        """build blocks"""
        for block_index, (
            input_channel,
            output_channel,
            block_base_channel,
            num_repeat,
            block_class,
        ) in enumerate(self.config):
            layers = nn.Sequential()
            for layer_index in range(num_repeat):

                layers.append(
                    ResBlock(
                        input_channel if layer_index == 0 else output_channel,
                        output_channel,
                        block_class(
                            input_channel=(
                                input_channel if layer_index == 0 else output_channel
                            ),
                            block_base_channel=block_base_channel,
                            stride=2 if layer_index == 0 else 1,
                            pre_activation=self.pre_activation,
                            widening_factor=self.widening_factor,
                        ),
                        stride=2 if layer_index == 0 else 1,
                    )
                )

            setattr(self, f"block{block_index+1}", layers)
