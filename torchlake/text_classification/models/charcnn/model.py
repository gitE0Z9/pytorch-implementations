import torch
from torch import nn
from torchlake.common.models import ConvBNReLU
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext

from .network import CharQuantization


class CharCNN(ModelBase):
    def __init__(
        self,
        char_size: int,
        hidden_dim: int = 256,
        classifier_dim: int = 1024,
        output_size: int = 1,
        dropout_prob: float = 0.5,
        context: NLPContext | None = None,
    ):
        """Character CNN in paper [1509.01626]

        Args:
            char_size (int): size of characters
            hidden_dim (int, optional): hidden dimension, Defaults to 256.
            classifier_dim (int, optional): classifier dimension, Defaults to 1024.
            output_size (int, optional): output size. Defaults to 1.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.context = context
        self.hidden_dim = hidden_dim
        self.classifier_dim = classifier_dim
        self.dropout_prob = dropout_prob
        super().__init__(char_size, output_size)

    @property
    def feature_dim(self) -> int:
        # paper page 3
        # l_6 = (l_0 - 96) / 3**3
        # 96 is from convolution layer
        # 3**3 if from three max pooling layer and each has the kernel size of 3
        # input dim = l_6 * frame_size
        return int((self.context.max_seq_len - 96) / 27 * self.hidden_dim)

    def build_foot(self, input_channel: int):
        self.foot = nn.Sequential(
            CharQuantization(input_channel, self.context),
        )

    def build_blocks(self):
        self.blocks = nn.Sequential(
            ConvBNReLU(
                self.input_channel,
                self.hidden_dim,
                7,
                enable_bn=False,
                dimension="1d",
            ),
            nn.MaxPool1d(3, 3),
            ConvBNReLU(
                self.hidden_dim,
                self.hidden_dim,
                7,
                enable_bn=False,
                dimension="1d",
            ),
            nn.MaxPool1d(3, 3),
            ConvBNReLU(
                self.hidden_dim,
                self.hidden_dim,
                3,
                enable_bn=False,
                dimension="1d",
            ),
            ConvBNReLU(
                self.hidden_dim,
                self.hidden_dim,
                3,
                enable_bn=False,
                dimension="1d",
            ),
            ConvBNReLU(
                self.hidden_dim,
                self.hidden_dim,
                3,
                enable_bn=False,
                dimension="1d",
            ),
            ConvBNReLU(
                self.hidden_dim,
                self.hidden_dim,
                3,
                enable_bn=False,
                dimension="1d",
            ),
            nn.MaxPool1d(3, 3),
        )

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.classifier_dim),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.classifier_dim, self.classifier_dim),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.classifier_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, h, s
        y = self.foot(x).transpose(-1, -2).float()
        #  b, h, s -> b*s, h
        y = self.blocks(y).transpose(-1, -2).contiguous().view(-1, self.feature_dim)

        # b*s, o
        return self.head(y)


def charCNN_small(
    char_size: int,
    output_size: int = 1,
    dropout_prob: float = 0.5,
    context: NLPContext | None = None,
):
    return CharCNN(
        char_size,
        hidden_dim=256,
        classifier_dim=1024,
        output_size=output_size,
        dropout_prob=dropout_prob,
        context=context,
    )


def charCNN_large(
    char_size: int,
    output_size: int = 1,
    dropout_prob: float = 0.5,
    context: NLPContext | None = None,
):
    return CharCNN(
        char_size,
        hidden_dim=1024,
        classifier_dim=2048,
        output_size=output_size,
        dropout_prob=dropout_prob,
        context=context,
    )
