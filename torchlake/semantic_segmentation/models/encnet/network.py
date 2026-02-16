import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.flatten import FlattenFeature


class EncodingModule2d(nn.Module):
    def __init__(self, input_channel: int, k: int = 32):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(k, input_channel))
        self.scales = nn.Parameter(torch.rand(k))
        self.stem = Conv2dNormActivation(input_channel, input_channel, 1)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_channel),
            nn.ReLU(True),
            FlattenFeature(dimension="1d"),
        )
        self.head = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(True)

    def forward(
        self,
        x: torch.Tensor,
        output_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): shape (B, D, H, W)

        Returns:
            torch.Tensor: _description_
        """
        B, D, _, _ = x.shape
        # B, N, D
        y: torch.Tensor = self.stem(x).view(B, D, -1).transpose(1, 2).contiguous()

        # scaled l2
        # B, N, K, D
        d: torch.Tensor = y[:, :, None, :] - self.centers[None, None, :, :]
        # B, N, K
        a = (self.scales[None, None, :] * d.pow(2).sum(-1)).softmax(-1)

        # aggregate
        # B, K, D
        y = torch.einsum("bnkd,bnk->bkd", d, a)

        # B, K, D -> B, D
        z = self.layers(y.transpose(1, 2).contiguous())

        y = self.head(z)
        y = self.activation(x + x * y[:, :, None, None])

        if output_latent:
            return y, z

        return y
