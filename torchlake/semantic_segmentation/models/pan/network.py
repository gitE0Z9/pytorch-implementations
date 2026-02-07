from itertools import pairwise
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.conv import ConvBNReLU


class FPA(nn.Module):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        hidden_dim_global: int,
        hidden_dims_local: Sequence[Sequence[int]],
        kernels: Sequence[int] = (7, 5, 3),
    ):
        """Feature Pyramid Attention

        Args:
            input_channel (int): input channel
            hidden_dim (int): hidden dimension of the 1x1 convolution in the main branch
            hidden_dim_global (int): hidden dimension of the 1x1 convolution in the global branch
            hidden_dims_local (Sequence[Sequence[int]]): hidden dimensions of convolution layers in the pyramid branch. elements are (d7x7, d5x5, d3x3) of the downsample branch then the upsample branch
            kernels (Sequence[int], optional): kernels of convolution layers in the pyramid branch. Defaults to (7, 5, 3).
        """
        super().__init__()
        assert (
            len(hidden_dims_local) == 2
        ), "only support two layers of the local branch"
        msg = "the number of kernels should match the number of hidden dimensions of a layer of the local branch"
        assert len(kernels) == len(hidden_dims_local[0]), msg
        assert len(kernels) == len(hidden_dims_local[1]), msg

        self.input_channel = input_channel
        self.hidden_dim = hidden_dim
        self.hidden_dim_global = hidden_dim_global
        self.hidden_dims_local = hidden_dims_local
        self.kernels = kernels

        self.branch = Conv2dNormActivation(input_channel, hidden_dim, 1)

        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2dNormActivation(input_channel, hidden_dim_global, 1),
        )

        self.branch_local_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    Conv2dNormActivation(in_c, out_c, kernel),
                )
                for (in_c, out_c), kernel in zip(
                    pairwise((input_channel, *hidden_dims_local[0])),
                    kernels,
                )
            ]
        )
        self.branch_local_neck = nn.ModuleList(
            [
                Conv2dNormActivation(in_c, out_c, kernel)
                for in_c, out_c, kernel in zip(
                    hidden_dims_local[0],
                    hidden_dims_local[1],
                    kernels,
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.branch(x)

        features = []
        z = x
        for block in self.branch_local_blocks:
            z = block(z)
            features.append(z)
        z = self.branch_local_neck[-1](features.pop())
        for block in self.branch_local_neck[::-1][1:]:
            zz: torch.Tensor = block(features.pop())
            z = zz + F.interpolate(z, size=zz.shape[2:])

        y = y * F.interpolate(z, size=y.shape[2:])
        y = y + self.branch_global(x)

        return y


class GAU(nn.Module):

    def __init__(
        self,
        input_channel_query: int,
        input_channel_key: int,
    ):
        """Global Attention Upsample

        Args:
            input_channel_query (int): input channel of query, i.e. low level feature maps
            input_channel_key (int): input channel of key, i.e. high level feature maps
            hidden_dim (int): hidden dimension of attented output
        """
        super().__init__()
        self.input_channel_query = input_channel_query
        self.input_channel_key = input_channel_key

        self.branch_query = nn.Sequential(
            Conv2dNormActivation(input_channel_query, input_channel_query, 3),
        )

        self.branch_key = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2dNormActivation(
                input_channel_key,
                input_channel_query,
                1,
                activation_layer=None,
            ),
            nn.Sigmoid(),
        )

        self.branch_reduction = nn.Sequential(
            Conv2dNormActivation(input_channel_key, input_channel_query, 1),
        )
        self.activation = nn.ReLU(True)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """forward method of GAU

        Args:
            q (torch.Tensor): query, i.e. low level feature maps
            k (torch.Tensor): key, i.e. high level feature maps

        Returns:
            torch.Tensor: output
        """
        a = self.branch_key(k) * self.branch_query(q)
        # return self.activation(a + F.interpolate(k, size=q.shape[2:]))
        return self.activation(
            a + F.interpolate(self.branch_reduction(k), size=q.shape[2:])
        )
