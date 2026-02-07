import torch
import torch.nn.functional as F
from torch import nn


class RCU(nn.Module):
    def __init__(self, input_channel: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_channel, input_channel, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(input_channel, input_channel, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class MultiResolutionFusion(nn.Module):
    def __init__(
        self,
        input_channel_shallow: int,
        input_channel_deep: int,
        output_channel: int,
    ):
        super().__init__()

        self.branch_shallow = nn.Conv2d(
            input_channel_shallow,
            output_channel,
            3,
            padding=1,
            bias=False,
        )

        self.branch_deep = nn.Conv2d(
            input_channel_deep,
            output_channel,
            3,
            padding=1,
            bias=False,
        )

    def forward(self, shallow_x: torch.Tensor, deep_x: torch.Tensor) -> torch.Tensor:
        return self.branch_shallow(shallow_x) + F.interpolate(
            self.branch_deep(deep_x),
            size=shallow_x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )


class ChainedResidualPooling(nn.Module):
    def __init__(self, input_channel: int, num_pooling: int = 4):
        super().__init__()
        self.num_pooling = num_pooling
        self.layers = nn.ModuleList(
            [
                nn.ReLU(True),
            ]
        )
        for _ in range(num_pooling):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channel, input_channel, 3, padding=1, bias=False),
                    nn.MaxPool2d(5, stride=1, padding=2),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers[0](x)
        z = y
        for l in range(1, self.num_pooling + 1):
            z = self.layers[l](z)
            y = y + z

        return y


class RefineNetBlock(nn.Module):
    def __init__(
        self,
        input_channel_shallow: int,
        input_channel_deep: int,
        output_channel: int,
    ):
        super().__init__()
        self.foot_shallow = nn.Sequential(
            nn.Conv2d(
                input_channel_shallow,
                output_channel,
                3,
                padding=1,
                bias=False,
            ),
            RCU(output_channel),
            RCU(output_channel),
        )

        if input_channel_deep > 0:
            self.foot_deep = nn.Sequential(
                RCU(input_channel_deep),
                RCU(input_channel_deep),
            )

            self.blocks = MultiResolutionFusion(
                output_channel,
                input_channel_deep,
                output_channel,
            )

        self.neck = ChainedResidualPooling(output_channel, 4)

        self.head = RCU(output_channel)

    def forward(
        self,
        shallow_x: torch.Tensor,
        deep_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.foot_shallow(shallow_x)
        if deep_x is not None:
            z = self.foot_deep(deep_x)
            y = self.blocks(y, z)
        y = self.neck(y)
        return self.head(y)
