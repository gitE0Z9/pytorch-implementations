import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation


class SEBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        pool_kernel_size: tuple[int] = (49, 49),
        pool_stride: tuple[int] = (16, 20),
    ):
        """SE block of upper branch Lite reduced ASPP in paper [1905.02244v5]

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
            pool_kernel_size (tuple[int], optional): kernel size of pool. Defaults to (49, 49).
            pool_stride (tuple[int], optional): stride of pool. Defaults to (16, 20).
        """
        super().__init__()
        self.block = Conv2dNormActivation(input_channel, output_channel, 1)
        self.weight = nn.Sequential(
            nn.AvgPool2d(pool_kernel_size, pool_stride),
            nn.Conv2d(input_channel, output_channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight(x)
        return F.interpolate(w, size=x.shape[2:], mode="bilinear") * self.block(x)


class LRASPP(nn.Module):

    def __init__(
        self,
        input_channels: list[int],
        hidden_dim: int,
        output_channel: int,
        pool_kernel_size: tuple[int] = (49, 49),
        pool_stride: tuple[int] = (16, 20),
    ):
        """Lite reduced ASPP in paper [1905.02244v5]

        Args:
            input_channels (list[int]): input channel
            hidden_dim (int): hidden dimension
            output_channel (int): output channel
            pool_kernel_size (tuple[int], optional): kernel size of pool. Defaults to (49, 49).
            pool_stride (tuple[int], optional): stride of pool. Defaults to (16, 20).
        """
        super().__init__()
        shallow_input_channel, deep_input_channel = input_channels
        self.block = nn.Sequential(
            # pool size should generate a 2x5 feature map
            # K1 + S1*(2-1) = H // 16 + 1
            # K2 + S2*(5-1) = W // 16 + 1
            SEBlock(deep_input_channel, hidden_dim, pool_kernel_size, pool_stride),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_dim, output_channel, 1),
        )
        self.shortcut = nn.Conv2d(shallow_input_channel, output_channel, 1)

    def forward(self, shallow_x: torch.Tensor, deep_x: torch.Tensor) -> torch.Tensor:
        return self.block(deep_x) + self.shortcut(shallow_x)
