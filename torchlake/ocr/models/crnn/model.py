import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.sequence_data.models.lstm import LSTMDiscriminator


class CRNN(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_size: int,
    ):
        """Convolution recurrent neural network, arxiv [1507.05717]

        Args:
            input_channel (int): input channel
            hidden_dim (int): rnn hidden dimension
            output_size (int): output size
        """
        self.hidden_dim = hidden_dim
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            # B, 64, 32, W
            Conv2dNormActivation(input_channel, 64, 3, norm_layer=None),
            # B, 64, 16, W//2
            nn.MaxPool2d(2, 2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            # B, 128, 16, W//2
            Conv2dNormActivation(64, 128, 3, norm_layer=None),
            # B, 128, 8, W//4
            nn.MaxPool2d(2, 2),
            # B, 256, 8, W//4
            Conv2dNormActivation(128, 256, 3, norm_layer=None),
            Conv2dNormActivation(256, 256, 3, norm_layer=None),
            # B, 256, 4, W//4
            nn.MaxPool2d((2, 1), (2, 1)),
            # B, 512, 4, W//4
            Conv2dNormActivation(256, 512, 3),
            Conv2dNormActivation(512, 512, 3),
            # B, 512, 2, W//4
            nn.MaxPool2d((2, 1), (2, 1)),
            # B, 512, 1, W//4-1
            Conv2dNormActivation(512, 512, 2, padding=0),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            LSTMDiscriminator(
                0,
                512,
                self.hidden_dim,
                output_size,
                num_layers=2,
                bidirectional=True,
                sequence_output=True,
                enable_embed=False,
                context=NlpContext(padding_idx=None),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor

        y = self.foot(x)
        y = self.blocks(y)
        _, _, h, _ = y.shape
        assert h == 1, "the height of conv must be 1"

        # b, w, c
        y = y.squeeze(2).transpose(1, 2)
        return self.head(y).transpose(0, 1)
