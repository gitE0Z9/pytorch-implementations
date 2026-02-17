import math

import torch
import torch.nn.functional as F
from torch import nn


class EMAttention2d(nn.Module):
    def __init__(
        self,
        input_channel: int,
        k: int = 64,
        lambda_a: float = 1,
        num_iter: int = 3,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.lambda_a = lambda_a
        self.momentum = momentum

        mu = torch.empty(input_channel, k)
        nn.init.normal_(mu, math.sqrt(2 / k))
        mu = F.normalize(mu, p=2, dim=1)
        self.mu = nn.Parameter(mu)

        self.stem = nn.Conv2d(input_channel, input_channel, 1)
        nn.init.normal_(self.stem.weight.data, math.sqrt(2 / input_channel))
        self.head = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_channel, input_channel, 1),
            nn.BatchNorm2d(input_channel),
        )
        nn.init.normal_(self.head[1].weight.data, math.sqrt(2 / input_channel))
        nn.init.ones_(self.head[2].weight.data)
        nn.init.zeros_(self.head[2].bias.data)

        self.activation = nn.ReLU(True)

    def forward(self, x: torch.Tensor, output_attention: bool = False) -> torch.Tensor:
        y: torch.Tensor = self.stem(x)

        with torch.no_grad():
            # b, n, c
            y = y.view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous()
            mu = self.mu[None, :, :].repeat(x.size(0), 1, 1)
            for _ in range(self.num_iter):
                # E step
                # b, n, c x b, c, k -> b, n, k
                z = torch.bmm(y, mu).mul(self.lambda_a).softmax(2)

                # M step
                # b, c, n x b, n, k -> b, c, k
                mu = torch.bmm(y.transpose(1, 2), z / (1e-6 + z.sum(1, keepdim=True)))
                mu = F.normalize(mu, p=2, dim=1)

            # update mu in MA way
            self.mu.copy_((self.momentum * self.mu + (1 - self.momentum) * mu).mean(0))

        # reestimation
        # b, c, k x b, k, n -> b, c, h, w
        y = torch.bmm(mu, z.transpose(1, 2).contiguous()).reshape(*x.shape)

        y = self.head(y)
        y = self.activation(y + x)

        if output_attention:
            return y, z

        return y
