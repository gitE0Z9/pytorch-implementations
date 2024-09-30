import torch
from torch import nn

from torchlake.common.models import MultiKernelConvModule


class TextCnn(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int = 100,
        output_size: int = 1,
        padding_idx: int | None = None,
        kernel_size: list[int] = [3, 4, 5],
        dropout_prob: float = 0.5,
    ):
        """TextCNN in paper[1408.5882]

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int, optional): dimension of convolution layer. Defaults to 100.
            output_size (int, optional): number of features of output. Defaults to 1.
            padding_idx (int | None, optional): index of padding token. Defaults to None.
            kernel_size (list[int], optional): size of kernels. Defaults to [3,4,5].
            dropout_prob (float, Defaults 0.5): dropout probability of fully connected layer. Defaults to 0.5.
        """
        super(TextCnn, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pool = MultiKernelConvModule(
            1,
            hidden_dim,
            [(k, embed_dim) for k in kernel_size],
            disable_padding=True,
            activation=nn.ReLU(inplace=True),
            reduction="max",
            concat_output=True,
        )
        self.fc = nn.Linear(hidden_dim * len(kernel_size), output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch_size, 1, Seq_len, embed_dim
        y = self.embed(x).unsqueeze(1)

        # Batch_size, filter_number * hidden_dim
        y = self.pool(y)

        y = self.dropout(y)

        # Batch_size, label_size
        y = self.fc(y)

        return y
