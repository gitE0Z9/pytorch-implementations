import torch
from torch import nn
from torchlake.common.models import KmaxPool1d


class WideConv1d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
    ):
        super(WideConv1d, self).__init__()
        self.conv = nn.Conv1d(
            input_channel,
            output_channel,
            kernel,
            padding=2 * (kernel // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DynamicKmaxPool1d(KmaxPool1d):

    def __init__(
        self,
        topk: int,
        seq_len: int,
        conv_layer_idx: int,
        conv_layer_total: int,
    ):
        """Dynamic top k max pooling 1d
        topk number is computed in paper [1404.2188v1]

        Args:
            topk (int): top k
            seq_len (int): sequence length
            conv_layer_idx (int): index of convolution layer
            conv_layer_total (int): total number of convolution layer
        """
        self.topk = max(topk, int((1 - conv_layer_idx / conv_layer_total) * seq_len))
        super(DynamicKmaxPool1d, self).__init__(self.topk)


class Block(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        topk: int,
        seq_len: int,
        conv_layer_idx: int,
        conv_layer_total: int,
    ):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            WideConv1d(input_channel, output_channel, kernel),
            DynamicKmaxPool1d(topk, seq_len, conv_layer_idx, conv_layer_total),
        )
        self.bias = nn.Parameter(torch.zeros((1, output_channel, 1)))
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y += self.bias
        return self.activation(y)


class Folding(nn.Module):

    def __init__(self):
        super(Folding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s = x.shape
        return x.reshape(b, c // 2, 2, s).sum(2)
