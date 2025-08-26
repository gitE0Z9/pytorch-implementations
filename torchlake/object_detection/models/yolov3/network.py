from typing import Sequence

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class RegHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_anchors: int,
        num_classes: int,
        coord_dims: int = 4,
    ):
        """last two layers of prediction head of YOLOv3

        Args:
            input_channel (int): input channel
            num_anchors (int): number of anchor boxes
            num_classes (int): number of classes
            coord_dims (int, optional): number of coordinate dimension. Defaults to 4.
        """
        self.num_anchors = num_anchors
        self.coord_dims = coord_dims
        self.num_classes = num_classes
        self.input_channel = input_channel
        self.output_size = num_anchors * (num_classes + coord_dims + 1)

        super().__init__()
        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                input_channel * 2,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.Conv2d(
                input_channel * 2,
                self.output_size,
                1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SPP(nn.Module):
    def __init__(
        self,
        kernel_sizes: Sequence[int | Sequence[int]] = (5, 9, 13),
    ):
        """SPP of yolov3

        output_channel = input_channel * (1 + len(self.kernel_sizes))

        Args:
            kernel_sizes (Sequence[int | Sequence[int]], optional): kernel sizes. Default to (5,9,13).
        """
        self.kernel_sizes = kernel_sizes

        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, *(layer(x) for layer in self.layers)), 1)
