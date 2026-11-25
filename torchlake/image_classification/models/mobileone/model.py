from typing import Literal, NamedTuple

from torch import nn


from ..repvgg import RepVGGBlock, RepVGG


class StageArgs(NamedTuple):
    base_dim: int
    width_multiplier: float
    num_block: int
    num_se_block: int
    num_branch: int

    @property
    def output_dim(self) -> int:
        return int(self.base_dim * self.width_multiplier)


class MobileOne(RepVGG):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: Literal["s0", "s1", "s2", "s3", "s4", "mu0", "mu1", "mu2"] = "s0",
        config: tuple[StageArgs] | None = None,
    ):
        """MobileOne [2206.04040]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            key (Literal["s0", "s1", "s2", "s3", "s4", "mu0", "mu1", "mu2"], optional): model variant, mu is the micro variant. Defaults to "s0".
            config (tuple[StageArgs] | None, optional): stage config, first stage is the stem. Defaults to None.
        """
        super().__init__(input_channel, output_size, key, config)

    @property
    def config(self) -> tuple[StageArgs]:
        # s: official specs: apple/ml-mobileone mobileone.py, the stem is capped at min(64, 64 * α)
        # mu: paper table 17, merged stages with the same width
        # the first stage is the stem, a single full 3x3 conv block with stride 2
        return (
            self._config
            or {
                "s0": (
                    # paper: StageArgs(64, 0.75, 1, 0, 4),
                    StageArgs(64, 0.75, 1, 0, 1),
                    StageArgs(64, 0.75, 2, 0, 4),
                    StageArgs(128, 1.0, 8, 0, 4),
                    StageArgs(256, 1.0, 10, 0, 4),
                    StageArgs(512, 2.0, 1, 0, 4),
                ),
                "s1": (
                    # paper: StageArgs(64, 1.5, 1, 0, 1),
                    StageArgs(64, 1.0, 1, 0, 1),
                    StageArgs(64, 1.5, 2, 0, 1),
                    StageArgs(128, 1.5, 8, 0, 1),
                    StageArgs(256, 2.0, 10, 0, 1),
                    StageArgs(512, 2.5, 1, 0, 1),
                ),
                "s2": (
                    # paper: StageArgs(64, 1.5, 1, 0, 1),
                    StageArgs(64, 1.0, 1, 0, 1),
                    StageArgs(64, 1.5, 2, 0, 1),
                    StageArgs(128, 2.0, 8, 0, 1),
                    StageArgs(256, 2.5, 10, 0, 1),
                    StageArgs(512, 4.0, 1, 0, 1),
                ),
                "s3": (
                    # paper: StageArgs(64, 2.0, 1, 0, 1),
                    StageArgs(64, 1.0, 1, 0, 1),
                    StageArgs(64, 2.0, 2, 0, 1),
                    StageArgs(128, 2.5, 8, 0, 1),
                    StageArgs(256, 3.0, 10, 0, 1),
                    StageArgs(512, 4.0, 1, 0, 1),
                ),
                "s4": (
                    # paper: StageArgs(64, 3.0, 1, 0, 1),
                    StageArgs(64, 1.0, 1, 0, 1),
                    StageArgs(64, 3.0, 2, 0, 1),
                    StageArgs(128, 3.5, 8, 0, 1),
                    StageArgs(256, 3.5, 10, 5, 1),
                    StageArgs(512, 4.0, 1, 1, 1),
                ),
                "mu0": (
                    StageArgs(64, 0.75, 1, 0, 3),
                    StageArgs(64, 0.75, 2, 0, 3),
                    StageArgs(128, 0.5, 4, 0, 3),
                    StageArgs(256, 0.5, 6, 0, 3),
                    StageArgs(512, 0.75, 1, 0, 3),
                ),
                "mu1": (
                    StageArgs(64, 0.75, 1, 0, 2),
                    StageArgs(64, 0.75, 2, 0, 2),
                    StageArgs(128, 0.75, 6, 0, 2),
                    StageArgs(256, 0.75, 8, 0, 2),
                    StageArgs(512, 1.0, 1, 0, 2),
                ),
                "mu2": (
                    StageArgs(64, 0.75, 1, 0, 2),
                    StageArgs(64, 0.75, 2, 0, 2),
                    StageArgs(128, 1.0, 6, 0, 2),
                    StageArgs(256, 1.0, 8, 0, 2),
                    StageArgs(512, 1.0, 1, 0, 2),
                ),
            }[self.key]
        )

    def build_foot(self, input_channel: int, **kwargs):
        stem_cfg = self.config[0]
        self.foot = RepVGGBlock(
            input_channel,
            stem_cfg.output_dim,
            3,
            stride=2,
            num_branch=stem_cfg.num_branch,
        )

    def build_blocks(self, **kwargs):
        in_c = self.config[0].output_dim

        self.blocks = nn.Sequential()
        for stage_args in self.config[1:]:
            stage = nn.Sequential()
            for block_idx in range(stage_args.num_block):
                stride = 2 if block_idx == 0 else 1
                # se will be applied to the last few blocks
                enable_se = block_idx >= stage_args.num_block - stage_args.num_se_block

                # depthwise
                stage.append(
                    RepVGGBlock(
                        in_c,
                        in_c,
                        3,
                        stride,
                        groups=in_c,
                        num_branch=stage_args.num_branch,
                        enable_se=enable_se,
                    )
                )
                # pointwise
                stage.append(
                    RepVGGBlock(
                        in_c,
                        stage_args.output_dim,
                        1,
                        num_branch=stage_args.num_branch,
                        enable_se=enable_se,
                    )
                )

                in_c = stage_args.output_dim

            self.blocks.append(stage)
