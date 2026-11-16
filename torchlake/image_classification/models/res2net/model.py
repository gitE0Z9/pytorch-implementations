from functools import partial
from typing import Literal

from ..resnet.model import BottleneckArgs, ResNet
from .network import BottleNeck


class Res2Net(ResNet):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        pre_activation: bool = False,
        key: Literal[50, 101, 152] = 50,
        config: tuple[BottleneckArgs] | None = None,
        split: int = 4,
        groups: int = 1,
    ):
        """Res2Net in paper [1904.01169v3]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            key (Literal[50, 101, 152], optional): key of configs. Defaults to 50.
            config (tuple[BottleneckArgs], optional): stage config, format is (input_dim, hidden_dim, expansion, number_block, layer_cls). Defaults to None.
            split (int, optional): s in paper, split input_channel into s * w channels. Defaults to 4.
            groups (int, optional): group of 3x3 filters. Defaults to 1.
        """
        self.split = split
        self.groups = groups
        super().__init__(
            input_channel,
            output_size,
            pre_activation=pre_activation,
            key=key,
            config=config,
        )

    @property
    def config(self) -> tuple[BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[BottleneckArgs]: parameters of layers
        """
        if self._config:
            return self._config

        layer_cls = partial(BottleNeck, split=self.split, groups=self.groups)
        return {
            50: (
                BottleneckArgs(64, 64, 4, 3, layer_cls),
                BottleneckArgs(256, 128, 4, 4, layer_cls),
                BottleneckArgs(512, 256, 4, 6, layer_cls),
                BottleneckArgs(1024, 512, 4, 3, layer_cls),
            ),
            101: (
                BottleneckArgs(64, 64, 4, 3, layer_cls),
                BottleneckArgs(256, 128, 4, 4, layer_cls),
                BottleneckArgs(512, 256, 4, 23, layer_cls),  # more block
                BottleneckArgs(1024, 512, 4, 3, layer_cls),
            ),
            152: (
                BottleneckArgs(64, 64, 4, 3, layer_cls),
                BottleneckArgs(256, 128, 4, 8, layer_cls),  # more block
                BottleneckArgs(512, 256, 4, 36, layer_cls),  # more block
                BottleneckArgs(1024, 512, 4, 3, layer_cls),
            ),
        }[self.key]
