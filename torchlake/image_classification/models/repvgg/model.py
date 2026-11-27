from copy import deepcopy
from typing import Literal, NamedTuple

from torch import nn

from torchlake.common.models.model_base import ModelBase

from .network import RepVGGBlock


class StageArgs(NamedTuple):
    base_dim: int
    width_multiplier: float
    num_block: int
    groups: int
    enable_se: bool

    @property
    def output_dim(self) -> int:
        return int(self.base_dim * self.width_multiplier)


class RepVGG(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: Literal[
            "a0",
            "a1",
            "a2",
            "b0",
            "b1",
            "b1g2",
            "b1g4",
            "b2",
            "b2g2",
            "b2g4",
            "b3",
            "b3g2",
            "b3g4",
            "d2se",
        ] = "a0",
        config: tuple[StageArgs] | None = None,
    ):
        """RepVGG [2101.03697]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            key (Literal["a0", "a1", "a2", "b0", "b1", "b1g2", "b1g4", "b2", "b2g2", "b2g4", "b3",  "b3g2", "b3g4", "d2se"], optional): model variant. Defaults to "a0".
            config (tuple[StageArgs] | None, optional): stage config, first stage is the stem. Defaults to None.
        """
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return self.config[-1].output_dim

    @property
    def config(self) -> tuple[StageArgs]:
        if self._config:
            return self._config

        # official specs: DingXiaoH/RepVGG repvgg.py, paper table 3
        # width of stage 2-5 is [64a, 128a, 256a, 512b], the stem is min(64, 64a), scaled down only
        # the first stage is the stem, a single block with stride 2
        return {
            "a0": (
                StageArgs(64, 0.75, 1, 1, False),
                StageArgs(64, 0.75, 2, 1, False),
                StageArgs(128, 0.75, 4, 1, False),
                StageArgs(256, 0.75, 14, 1, False),
                StageArgs(512, 2.5, 1, 1, False),
            ),
            "a1": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 1.0, 2, 1, False),
                StageArgs(128, 1.0, 4, 1, False),
                StageArgs(256, 1.0, 14, 1, False),
                StageArgs(512, 2.5, 1, 1, False),
            ),
            "a2": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 1.5, 2, 1, False),
                StageArgs(128, 1.5, 4, 1, False),
                StageArgs(256, 1.5, 14, 1, False),
                StageArgs(512, 2.75, 1, 1, False),
            ),
            "b0": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 1.0, 4, 1, False),
                StageArgs(128, 1.0, 6, 1, False),
                StageArgs(256, 1.0, 16, 1, False),
                StageArgs(512, 2.5, 1, 1, False),
            ),
            "b1": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 2.0, 4, 1, False),
                StageArgs(128, 2.0, 6, 1, False),
                StageArgs(256, 2.0, 16, 1, False),
                StageArgs(512, 4.0, 1, 1, False),
            ),
            "b1g2": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 2.0, 4, 2, False),
                StageArgs(128, 2.0, 6, 2, False),
                StageArgs(256, 2.0, 16, 2, False),
                StageArgs(512, 4.0, 1, 2, False),
            ),
            "b1g4": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 2.0, 4, 4, False),
                StageArgs(128, 2.0, 6, 4, False),
                StageArgs(256, 2.0, 16, 4, False),
                StageArgs(512, 4.0, 1, 4, False),
            ),
            "b2": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 2.5, 4, 1, False),
                StageArgs(128, 2.5, 6, 1, False),
                StageArgs(256, 2.5, 16, 1, False),
                StageArgs(512, 5.0, 1, 1, False),
            ),
            "b2g2": (
                StageArgs(64, 1.0, 1, 2, False),
                StageArgs(64, 2.5, 4, 2, False),
                StageArgs(128, 2.5, 6, 2, False),
                StageArgs(256, 2.5, 16, 2, False),
                StageArgs(512, 5.0, 1, 2, False),
            ),
            "b2g4": (
                StageArgs(64, 1.0, 1, 4, False),
                StageArgs(64, 2.5, 4, 4, False),
                StageArgs(128, 2.5, 6, 4, False),
                StageArgs(256, 2.5, 16, 4, False),
                StageArgs(512, 5.0, 1, 4, False),
            ),
            "b3": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 3.0, 4, 1, False),
                StageArgs(128, 3.0, 6, 1, False),
                StageArgs(256, 3.0, 16, 1, False),
                StageArgs(512, 5.0, 1, 1, False),
            ),
            "b3g2": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 3.0, 4, 2, False),
                StageArgs(128, 3.0, 6, 2, False),
                StageArgs(256, 3.0, 16, 2, False),
                StageArgs(512, 5.0, 1, 2, False),
            ),
            "b3g4": (
                StageArgs(64, 1.0, 1, 1, False),
                StageArgs(64, 3.0, 4, 4, False),
                StageArgs(128, 3.0, 6, 4, False),
                StageArgs(256, 3.0, 16, 4, False),
                StageArgs(512, 5.0, 1, 4, False),
            ),
            "d2se": (
                StageArgs(64, 1.0, 1, 1, True),
                StageArgs(64, 2.5, 8, 1, True),
                StageArgs(128, 2.5, 14, 1, True),
                StageArgs(256, 2.5, 24, 1, True),
                StageArgs(512, 5.0, 1, 1, True),
            ),
        }[self.key]

    def build_foot(self, input_channel: int, **kwargs):
        stem_cfg = self.config[0]
        self.foot = RepVGGBlock(
            input_channel,
            stem_cfg.output_dim,
            3,
            stride=2,
            enable_se=stem_cfg.enable_se,
        )

    def build_blocks(self, **kwargs):
        in_c = self.config[0].output_dim

        # official code applies groups to every other layer, counted across stages from 1
        layer_idx = 1
        self.blocks = nn.Sequential()
        for stage_cfg in self.config[1:]:
            stage = nn.Sequential()
            for block_idx in range(stage_cfg.num_block):
                stage.append(
                    RepVGGBlock(
                        in_c,
                        stage_cfg.output_dim,
                        3,
                        stride=2 if block_idx == 0 else 1,
                        groups=stage_cfg.groups if layer_idx % 2 == 0 else 1,
                        enable_se=stage_cfg.enable_se,
                    )
                )

                in_c = stage_cfg.output_dim
                layer_idx += 1

            self.blocks.append(stage)

    def reparameterize(self) -> "RepVGG":
        """fuse branches of every block into a single conv, for inference only

        Returns:
            RepVGG: new reparameterized model
        """
        dest = deepcopy(self)
        for block in [dest.foot, *(block for stage in dest.blocks for block in stage)]:
            block.layers[0] = block.layers[0].reparameterize()

        return dest
