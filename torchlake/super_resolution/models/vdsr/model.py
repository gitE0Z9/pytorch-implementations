from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock


class VDSR(ModelBase):
    def __init__(
        self,
        input_channel: int = 1,
        hidden_dim: int = 64,
        num_blocks: int = 20,
    ):
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        super().__init__(input_channel, input_channel)

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def build_foot(self, input_channel, **kwargs):
        blocks = nn.Sequential(
            nn.Conv2d(input_channel, self.hidden_dim, 3, padding=1), nn.ReLU(True)
        )
        for _ in range(self.num_blocks - 2):
            blocks.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1))
            blocks.append(nn.ReLU(True))
        blocks.append(nn.Conv2d(self.hidden_dim, input_channel, 3, padding=1))

        self.foot = ResBlock(
            input_channel,
            input_channel,
            block=blocks,
            activation=None,
        )

    def build_head(self, _, **kwargs):
        self.head = nn.Identity()
