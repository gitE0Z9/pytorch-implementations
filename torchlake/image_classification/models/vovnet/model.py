from typing import Literal, NamedTuple

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock

from .network import OSAModule


class StageArgs(NamedTuple):
    hidden_dim: int
    output_dim: int
    num_block: int
    num_layer: int


class VoVNet(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: Literal["27_slim", 39, 57] = 39,
        config: tuple[StageArgs] | None = None,
    ):
        """VoVNet [1904.09730]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            key (Literal["27_slim", 39, 57], optional): model variant. Defaults to 39.
            config (tuple[StageArgs] | None, optional): stage config. Defaults to None.
        """
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return self.config[-1].output_dim

    @property
    def config(self) -> tuple[StageArgs]:
        # official specs: stigma0617/VoVNet.pytorch models_vovnet/vovnet.py
        return (
            self._config
            or {
                "27_slim": (
                    StageArgs(64, 128, 1, 5),
                    StageArgs(80, 256, 1, 5),
                    StageArgs(96, 384, 1, 5),
                    StageArgs(112, 512, 1, 5),
                ),
                39: (
                    StageArgs(128, 256, 1, 5),
                    StageArgs(160, 512, 1, 5),
                    StageArgs(192, 768, 2, 5),
                    StageArgs(224, 1024, 2, 5),
                ),
                57: (
                    StageArgs(128, 256, 1, 5),
                    StageArgs(160, 512, 1, 5),
                    StageArgs(192, 768, 4, 5),
                    StageArgs(224, 1024, 3, 5),
                ),
            }[self.key]
        )

    def build_foot(self, input_channel: int, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 3, stride=2),
            Conv2dNormActivation(64, 64, 3),
            Conv2dNormActivation(64, 128, 3, stride=2),
        )

    def build_blocks(self, **kwargs):
        in_c = self.foot[-1].out_channels

        self.blocks = nn.Sequential()
        for stage_idx, stage_cfg in enumerate(self.config):
            stage = nn.Sequential()
            # the first stage is already downsampled by the stem
            if stage_idx > 0:
                stage.append(nn.MaxPool2d(3, 2, ceil_mode=True))

            for _ in range(stage_cfg.num_block):
                block = OSAModule(
                    in_c,
                    stage_cfg.hidden_dim,
                    stage_cfg.output_dim,
                    stage_cfg.num_layer,
                )
                # paper has no identity, official code adds it to repeated modules
                if in_c == stage_cfg.output_dim:
                    block = ResBlock(
                        in_c,
                        stage_cfg.output_dim,
                        block=block,
                        activation=None,
                    )
                stage.append(block)

                in_c = stage_cfg.output_dim

            self.blocks.append(stage)
