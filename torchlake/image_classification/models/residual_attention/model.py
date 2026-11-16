# https://github.com/fwang91/residual-attention-network
from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

from .network import AttentionModule, build_residual_unit


class ResidualAttentionNetwork(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: Literal[56, 92] = 56,
        config: tuple[tuple[int]] | None = None,
    ):
        """Residual attention network [1704.06904v1]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            key (Literal[56, 92], optional): key of configs. Defaults to 56.
            config (tuple[tuple[int]], optional): stage config, format is (input_channel, output_channel, num_residual_unit, num_attention_module, num_skip). Defaults to None.
        """
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return self.config[-1][1]

    @property
    def config(self) -> tuple[tuple[int]]:
        # input_channel, output_channel, num_residual_unit, num_attention_module, num_skip
        return (
            self._config
            or {
                56: (
                    (64, 256, 1, 1, 2),
                    (256, 512, 1, 1, 1),
                    (512, 1024, 1, 1, 0),
                    (1024, 2048, 3, 0, 0),
                ),
                92: (
                    (64, 256, 1, 1, 2),
                    (256, 512, 1, 2, 1),
                    (512, 1024, 1, 3, 0),
                    (1024, 2048, 3, 0, 0),
                ),
            }[self.key]
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                self.config[0][0],
                7,
                stride=2,
                norm_layer=None,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(self.config[0][0]),
            nn.ReLU(True),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for stage_idx, (
            in_c,
            out_c,
            num_residual_unit,
            num_attention_module,
            num_skip,
        ) in enumerate(self.config):
            stage = nn.Sequential()
            # residual units, the first one downsamples except the first stage
            for i in range(num_residual_unit):
                stage.append(
                    build_residual_unit(
                        in_c if i == 0 else out_c,
                        out_c,
                        stride=2 if i == 0 and stage_idx > 0 else 1,
                    )
                )
            # attention modules
            for _ in range(num_attention_module):
                stage.append(
                    AttentionModule(out_c, out_c, p=1, t=2, r=1, num_skip=num_skip)
                )

            self.blocks.append(stage)

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        # residual units are pre-activated, activate the last output
        self.head.insert(0, nn.BatchNorm2d(self.feature_dim))
        self.head.insert(1, nn.ReLU(True))
