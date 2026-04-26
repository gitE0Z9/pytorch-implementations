import torch
from torch import nn


class DownSampling(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        enable_in: bool = True,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.enable_in = enable_in
        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_channel, output_channel, 4, 2, 1, bias=not enable_in),
        )

        # use in instead of bn, since in the paper, they use batch size = 1 to mimic in
        if enable_in:
            self.layers.append(
                nn.InstanceNorm2d(output_channel),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpSampling(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(input_channel, output_channel, 4, 2, 1, bias=False),
            # use in instead of bn, since in the paper, they use batch size = 1 to mimic in
            nn.InstanceNorm2d(output_channel),
        )

        if dropout_prob > 0:
            self.layers.append(nn.Dropout(dropout_prob))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.layers(x), z], dim=1)
