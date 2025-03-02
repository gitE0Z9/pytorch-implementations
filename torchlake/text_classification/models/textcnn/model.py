import torch
from torch import nn

from torchlake.common.models import MultiKernelConvModule
from torchlake.common.models.model_base import ModelBase


class TextCNN(ModelBase):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int = 100,
        output_size: int = 1,
        padding_idx: int | None = None,
        kernels: list[int] = [3, 4, 5],
        dropout_prob: float = 0.5,
    ):
        """TextCNN in paper[1408.5882]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int, optional): dimension of convolution layer. Defaults to 100.
            output_size (int, optional): number of features of output. Defaults to 1.
            padding_idx (int | None, optional): index of padding token. Defaults to None.
            kernels (list[int], optional): size of kernels. Defaults to [3,4,5].
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.kernels = kernels
        self.dropout_prob = dropout_prob
        super().__init__(vocab_size, output_size)

    @property
    def feature_dim(self) -> int:
        return self.hidden_dim * len(self.kernels)

    def build_foot(self, vocab_size: int):
        self.foot = nn.Embedding(
            vocab_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

    def build_blocks(self):
        self.blocks = MultiKernelConvModule(
            1,
            self.hidden_dim,
            [(k, self.embed_dim) for k in self.kernels],
            disable_padding=True,
            activation=nn.ReLU(inplace=True),
            reduction="max",
            concat_output=True,
        )

    def build_head(self, output_size):
        self.head = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.feature_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch_size, 1, Seq_len, embed_dim
        y = self.foot(x).unsqueeze(1)

        # Batch_size, filter_number * hidden_dim
        y = self.blocks(y)

        # Batch_size, label_size
        return self.head(y)
