from typing import Sequence

import torch
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.model_base import ModelBase
from torchlake.semantic_segmentation.models.fcn.network import (
    init_deconv_with_bilinear_kernel,
)

from .network import DenseBlock, TransitionDown


class FCDenseNet(ModelBase):
    def __init__(
        self,
        input_channel,
        output_size,
        growth_rate: int,
        hidden_dim_stem: int,
        num_layers_symmetric: Sequence[int],
        num_layer_middle: int,
        dropout_prob: float = 0.2,
    ):
        self.growth_rate = growth_rate
        self.hidden_dim_stem = hidden_dim_stem
        self.num_layers_symmetric = num_layers_symmetric
        self.num_layer_middle = num_layer_middle
        self.dropout_prob = dropout_prob
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            nn.Conv2d(input_channel, self.hidden_dim_stem, 3, padding=1),
        )

        nn.init.kaiming_uniform_(self.foot[0].weight.data, nonlinearity="relu")

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()

        in_c = self.hidden_dim_stem
        for num_layer in self.num_layers_symmetric:
            h = num_layer * self.growth_rate
            self.blocks.append(
                DenseBlock(
                    in_c,
                    self.growth_rate,
                    num_layer=num_layer,
                    dropout_prob=self.dropout_prob,
                )
            )
            self.blocks.append(
                TransitionDown(in_c + h, dropout_prob=self.dropout_prob),
            )

            in_c += h

        self.blocks.append(
            DenseBlock(
                in_c,
                self.growth_rate,
                num_layer=self.num_layer_middle,
                dropout_prob=self.dropout_prob,
            )
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList()

        in_c = self.growth_rate * self.num_layer_middle
        down_c = self.hidden_dim_stem + self.growth_rate * sum(
            self.num_layers_symmetric
        )
        for num_layer in self.num_layers_symmetric[::-1]:
            h = self.growth_rate * num_layer
            self.neck.append(
                nn.ConvTranspose2d(in_c, h, 4, stride=2, padding=1),
                # nn.ConvTranspose2d(in_c, h, 4, stride=2, padding=1, bias=False),
            )
            self.neck.append(
                DenseBlock(
                    down_c + h,
                    self.growth_rate,
                    num_layer=num_layer,
                    dropout_prob=self.dropout_prob,
                )
            )
            down_c -= self.growth_rate * num_layer
            in_c = h

        for block in self.neck:
            if isinstance(block, nn.ConvTranspose2d):
                # init_deconv_with_bilinear_kernel(block)
                nn.init.kaiming_uniform_(block.weight.data, nonlinearity="relu")

    def build_head(self, output_size, **kwargs):
        in_c = (
            self.hidden_dim_stem + 3 * self.num_layers_symmetric[0] * self.growth_rate
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_c, output_size, 1),
        )
        nn.init.kaiming_uniform_(self.head[0].weight.data, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        features = []
        for i, block in enumerate(self.blocks[:-1]):
            y = block(y)

            if i % 2 == 0:
                features.append(y)
        y = self.blocks[-1](y)

        up_path_num_layers = self.num_layers_symmetric[::-1]
        y = y[:, : self.growth_rate * self.num_layer_middle]
        for i, neck in enumerate(self.neck[:-1]):
            y = neck(y)

            if i % 2 == 0:
                cropper = CenterCrop(y.shape[2:])
                z = features.pop()
                y = torch.cat((y, cropper(z)), 1)
            else:
                y = y[:, : self.growth_rate * up_path_num_layers[i // 2]]
        y = self.neck[-1](y)

        return self.head(y)


def fc_densenet_56(
    input_channel: int = 3,
    output_size: int = 1,
    dropout_prob: float = 0.2,
) -> FCDenseNet:
    return FCDenseNet(
        input_channel,
        output_size,
        growth_rate=12,
        hidden_dim_stem=48,
        num_layers_symmetric=(4, 4, 4, 4, 4),
        num_layer_middle=4,
        dropout_prob=dropout_prob,
    )


def fc_densenet_67(
    input_channel: int = 3,
    output_size: int = 1,
    dropout_prob: float = 0.2,
) -> FCDenseNet:
    return FCDenseNet(
        input_channel,
        output_size,
        growth_rate=16,
        hidden_dim_stem=48,
        num_layers_symmetric=(5, 5, 5, 5, 5),
        num_layer_middle=5,
        dropout_prob=dropout_prob,
    )


def fc_densenet_103(
    input_channel: int = 3,
    output_size: int = 1,
    dropout_prob: float = 0.2,
) -> FCDenseNet:
    return FCDenseNet(
        input_channel,
        output_size,
        growth_rate=16,
        hidden_dim_stem=48,
        num_layers_symmetric=(4, 5, 7, 10, 12),
        num_layer_middle=15,
        dropout_prob=dropout_prob,
    )
