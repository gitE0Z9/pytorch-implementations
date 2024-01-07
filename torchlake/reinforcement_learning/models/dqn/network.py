import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class QNetwork(nn.Module):
    def __init__(self, action_size: int):
        super(QNetwork, self).__init__()
        self.block1 = ConvBnRelu(1, 16, 8, 4)
        self.block2 = ConvBnRelu(16, 32, 4, 2)  # 32 * 9 * 9
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block1(x)
        y = self.block2(y)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        return y
