from typing import Any

from torch import nn
from torchlake.common.models.model_base import ModelBase

from .network import DiracConv2d


class DiracNet(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: int = 1,
        depth_multiplier: int = 1,
    ):
        """_summary_

        Args:
            input_channel (int, optional): _description_. Defaults to 3.
            output_size (int, optional): _description_. Defaults to 1.
            width_multiplier (int, optional): k in paper. Defaults to 1.
            depth_multiplier (int, optional): N in paper. Defaults to 1.
        """
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return int(64 * self.width_multiplier)

    @property
    def config(self) -> list[list[Any]]:
        return []

    def build_foot(self, input_channel):
        self.foot = DiracConv2d(input_channel, 16, 3)

    def build_blocks(self):
        blocks = []

        C = int(16 * self.width_multiplier)
        L = int(2 * self.depth_multiplier)
        NUM_STAGE = 3
        for stage_idx in range(NUM_STAGE):
            d = C * (2**stage_idx)
            for block_idx in range(L):
                # stage idx 0: d->d
                # stage idx 1: d//2->d, d->d
                # stage idx 2: d//2->d, d->d
                blocks.append(
                    DiracConv2d(
                        d // 2 if stage_idx > 0 and block_idx == 0 else d,
                        d,
                        3,
                    )
                )

            if stage_idx < NUM_STAGE - 1:
                blocks.append(nn.MaxPool2d(2, 2))

        self.blocks = nn.Sequential(*blocks)
