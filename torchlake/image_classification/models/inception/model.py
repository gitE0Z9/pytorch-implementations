import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

from ..alexnet.network import LocalResponseNorm
from .network import AuxiliaryClassifier, InceptionBlock


class Inception(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.4,
        aux_dropout_prob: float = 0.4,
    ):
        self.dropout_prob = dropout_prob
        self.aux_dropout_prob = aux_dropout_prob
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 1024

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 7, stride=2, norm_layer=None),
            nn.MaxPool2d(3, stride=2, padding=1),
            LocalResponseNorm(),
            Conv2dNormActivation(64, 64, 1, norm_layer=None),
            Conv2dNormActivation(64, 192, 3, norm_layer=None),
            LocalResponseNorm(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            InceptionBlock(192, (64, (96, 128), (16, 32), 32)),
            InceptionBlock(256, (128, (128, 192), (32, 96), 64)),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(480, (192, (96, 208), (16, 48), 64)),
            # aux clf here
            InceptionBlock(512, (160, (112, 224), (24, 64), 64)),
            InceptionBlock(512, (128, (128, 256), (24, 64), 64)),
            InceptionBlock(512, (112, (144, 288), (32, 64), 64)),
            # aux clf here
            InceptionBlock(528, (256, (160, 320), (32, 128), 128)),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(832, (256, (160, 320), (32, 128), 128)),
            InceptionBlock(832, (384, (192, 384), (48, 128), 128)),
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                AuxiliaryClassifier(
                    512,
                    self.output_size,
                    hidden_dims=(128, 1024),
                    dropout_prob=self.aux_dropout_prob,
                ),
                AuxiliaryClassifier(
                    528,
                    self.output_size,
                    hidden_dims=(128, 1024),
                    dropout_prob=self.aux_dropout_prob,
                ),
            ]
        )

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(1, nn.Dropout(p=self.dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        y = self.foot(x)

        if self.training:
            outputs = []

            y = self.blocks[:4](y)
            outputs.append(self.neck[0](y))

            y = self.blocks[4:7](y)
            outputs.append(self.neck[1](y))

            y = self.blocks[7:](y)
            outputs.append(self.head(y))

            return outputs
        else:
            y = self.blocks(y)
            return self.head(y)


class InceptionV2(ModelBase): ...
