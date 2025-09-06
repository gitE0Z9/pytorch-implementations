import torch
from torch import nn
from torchlake.common.models import ConvBnRelu
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext

from .network import CharQuantization


class CharCNN(ModelBase):
    def __init__(
        self,
        char_size: int,
        output_size: int = 1,
        dropout_prob: float = 0.5,
        context: NlpContext = NlpContext(),
    ):
        """Character CNN in paper [1509.01626]

        Args:
            char_size (int): size of characters
            output_size (int, optional): output size. Defaults to 1.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            context (NlpContext, optional): . Defaults to NlpContext().
        """
        self.context = context
        self.dropout_prob = dropout_prob
        super().__init__(char_size, output_size)

    @property
    def feature_dim(self) -> int:
        # paper page 3
        # l_6 = (l_0 - 96) / 3**3
        # input dim = l_6 * frame_size
        return int((self.context.max_seq_len - 96) / 27 * 256)

    def build_foot(self, input_channel: int):
        self.foot = nn.Sequential(
            CharQuantization(input_channel, self.context),
        )

    def build_blocks(self):
        self.blocks = nn.Sequential(
            ConvBnRelu(self.input_channel, 256, 7, enable_bn=False, dimension="1d"),
            nn.MaxPool1d(3, 3),
            ConvBnRelu(256, 256, 7, enable_bn=False, dimension="1d"),
            nn.MaxPool1d(3, 3),
            ConvBnRelu(256, 256, 3, enable_bn=False, dimension="1d"),
            ConvBnRelu(256, 256, 3, enable_bn=False, dimension="1d"),
            ConvBnRelu(256, 256, 3, enable_bn=False, dimension="1d"),
            ConvBnRelu(256, 256, 3, enable_bn=False, dimension="1d"),
            nn.MaxPool1d(3, 3),
        )

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.Dropout(self.dropout_prob),
            nn.Linear(1024, 1024),
            nn.Dropout(self.dropout_prob),
            nn.Linear(1024, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, h, s
        y = self.foot(x).transpose(-1, -2).float()
        #  b, h, s -> b*s, h
        y = self.blocks(y).transpose(-1, -2).contiguous().view(-1, self.feature_dim)

        # b*s, o
        return self.head(y)
