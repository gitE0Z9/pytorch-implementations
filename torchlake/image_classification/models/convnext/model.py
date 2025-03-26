from typing import Literal

from torch import nn
from torchlake.common.models import ResBlock
from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.model_base import ModelBase

from .network import BottleNeck


class ConvNeXt(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        hidden_dim: int | None = None,
        num_layers: list[int] | None = None,
        size: Literal["tiny", "small", "base", "large"] | None = None,
    ):
        assert size is not None or (
            hidden_dim is not None and num_layers is not None
        ), "must provide custom spec or choose one of default spec"
        self.size = size

        cfg = self.config
        self.num_layers = num_layers or cfg[0]
        self.hidden_dim = hidden_dim or cfg[1]
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return int(self.hidden_dim * 8)

    @property
    def config(self) -> tuple[list[int], int]:
        # num_block, hidden_dim
        return {
            "tiny": ([3, 3, 9, 3], 96),
            "small": ([3, 3, 27, 3], 96),
            "base": ([3, 3, 27, 3], 128),
            "large": ([3, 3, 27, 3], 192),
        }.get(self.size, None)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            nn.Conv2d(input_channel, self.hidden_dim, 4, stride=4),
            nn.GroupNorm(1, self.hidden_dim),
        )

    def build_blocks(self, **kwargs):
        blocks = []

        hidden_dim = self.hidden_dim
        for stage_idx, num_block in enumerate(self.num_layers):
            for _ in range(num_block):
                blocks.append(
                    ResBlock(
                        hidden_dim,
                        hidden_dim,
                        block=BottleNeck(hidden_dim),
                        activation=None,
                    ),
                )

            if stage_idx != (len(self.num_layers) - 1):
                blocks.extend(
                    [
                        nn.GroupNorm(1, hidden_dim),
                        nn.Conv2d(
                            hidden_dim,
                            hidden_dim * 2,
                            2,
                            stride=2,
                        ),
                    ]
                )

            hidden_dim *= 2

        self.blocks = nn.Sequential(*blocks)

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            FlattenFeature(start_dim=-1),  # no flatten
            nn.GroupNorm(1, self.feature_dim),
            FlattenFeature(reduction=None),  # no reduction
            nn.Linear(self.feature_dim, output_size),
        )
