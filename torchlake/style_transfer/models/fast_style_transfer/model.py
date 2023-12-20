from torch import nn
from .network import ConvBlock, ResidualBlock


class FastStyleTransfer(nn.Module):
    def __init__(self, num_block: int = 5):
        super(FastStyleTransfer, self).__init__()
        self.initial_conv = nn.Sequential(
            ConvBlock(3, 32, 9),
            ConvBlock(32, 64, 3, stride=2),
            ConvBlock(64, 128, 3, stride=2),
        )
        self.res_blocks = nn.Sequential(*([ResidualBlock(128)] * num_block))
        self.upsampling = nn.Sequential(
            ConvBlock(128, 64, 3, enable_deconv=True),
            ConvBlock(64, 32, 3, enable_deconv=True),
            ConvBlock(32, 3, 9, enable_in=False, enable_relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.initial_conv(x)
        y = self.res_blocks(y)
        y = self.upsampling(y)
        # y = y.tanh()
        return y
