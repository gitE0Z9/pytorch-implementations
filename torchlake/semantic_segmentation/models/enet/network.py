import torch
from torch import nn
import torch.nn.functional as F

from torchlake.common.models.conv import ConvBNReLU


class Stem(nn.Module):
    def __init__(self, input_channel: int, hidden_dim: int):
        super().__init__()
        self.input_channel = input_channel
        self.branch1 = nn.Conv2d(input_channel, hidden_dim, 3, stride=2, padding=1)
        self.branch2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch1(x), self.branch2(x)), 1)


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        reduction_ratio: float = 4,
        stride: int = 1,
        dilation: int = 1,
        dropout_prob: float = 0.1,
        asymmetric: bool = False,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_dim = output_channel // reduction_ratio
        self.stride = stride
        self.dilation = dilation
        self.reduction_ratio = reduction_ratio
        self.dropout_prob = dropout_prob

        block = nn.Sequential(
            ConvBNReLU(
                input_channel,
                self.hidden_dim,
                1 if stride == 1 else 2,
                stride=stride,
                activation=nn.PReLU(self.hidden_dim),
            ),
            # main conv is here
            ConvBNReLU(self.hidden_dim, output_channel, 1, activation=None),
            nn.Dropout2d(dropout_prob),
        )

        if asymmetric:
            block.insert(
                1,
                ConvBNReLU(
                    self.hidden_dim,
                    self.hidden_dim,
                    (5, 1),
                    padding=(2, 0),
                    activation=nn.PReLU(self.hidden_dim),
                ),
            )
            block.insert(
                2,
                ConvBNReLU(
                    self.hidden_dim,
                    self.hidden_dim,
                    (1, 5),
                    padding=(0, 2),
                    activation=nn.PReLU(self.hidden_dim),
                ),
            )
        else:
            block.insert(
                1,
                ConvBNReLU(
                    self.hidden_dim,
                    self.hidden_dim,
                    3,
                    padding=(3 + 2 * (dilation - 1)) // 2,
                    dilation=dilation,
                    activation=nn.PReLU(self.hidden_dim),
                ),
            )

        self.block = block
        if stride > 1:
            self.shortcut = nn.MaxPool2d(2, 2, return_indices=True)
        self.activation = nn.PReLU(output_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if hasattr(self, "shortcut"):
            z, indices = self.shortcut(x)
            if self.input_channel < self.output_channel:
                z = F.pad(z, (0, 0, 0, 0, 0, self.output_channel - self.input_channel))
            y = y + z
            return self.activation(y), indices
        else:
            y = y + x
            return self.activation(y)


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        reduction_ratio: float = 4,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_dim = output_channel // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.dropout_prob = dropout_prob

        self.block = nn.Sequential(
            ConvBNReLU(
                input_channel,
                self.hidden_dim,
                1,
                activation=nn.PReLU(self.hidden_dim),
            ),
            ConvBNReLU(
                self.hidden_dim,
                self.hidden_dim,
                4,
                stride=2,
                padding=1,
                activation=nn.PReLU(self.hidden_dim),
                deconvolution=True,
            ),
            ConvBNReLU(self.hidden_dim, output_channel, 1, activation=None),
            nn.Dropout2d(dropout_prob),
        )

        self.shortcut = nn.ModuleList(
            [
                nn.Conv2d(self.input_channel, self.output_channel, 1, bias=False),
                nn.MaxUnpool2d(2, 2),
            ]
        )

        self.activation = nn.PReLU(output_channel)

    def forward(self, x: torch.Tensor, pooling_indices: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if hasattr(self, "shortcut"):
            z = self.shortcut[0](x)
            z = self.shortcut[1](z, pooling_indices)
            y = y + z
        else:
            y = y + x

        return self.activation(y)
