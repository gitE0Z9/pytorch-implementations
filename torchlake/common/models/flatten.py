from typing import Literal

import torch
from torch import nn


class FlattenFeature(nn.Module):

    def __init__(
        self,
        reduction: Literal["mean", "max"] | None = "mean",
        dimension: Literal["1d", "2d", "3d"] = "2d",
        start_dim: int = 1,
        end_dim: int = -1,
    ):
        """_summary_

        Args:
            reduction (Literal["mean", "max"] | None, optional): redution mode. Defaults to "mean".
            dimension (Literal["1d", "2d", "3d"], optional): 1d, 2d or 3d. Defaults to "2d".
            start_dim: first dim to flatten. Defaults to 1.
            end_dim: first dim to flatten. Defaults to -1.
        """
        super(FlattenFeature, self).__init__()
        pooling_layer = (
            {
                "mean": {
                    "1d": nn.AdaptiveAvgPool1d((1,)),
                    "2d": nn.AdaptiveAvgPool2d((1, 1)),
                    "3d": nn.AdaptiveAvgPool3d((1, 1, 1)),
                },
                "max": {
                    "1d": nn.AdaptiveMaxPool1d((1,)),
                    "2d": nn.AdaptiveMaxPool2d((1, 1)),
                    "3d": nn.AdaptiveMaxPool3d((1, 1, 1)),
                },
            }
            .get(reduction, {})
            .get(dimension, nn.Identity())
        )

        self.pool = nn.Sequential(
            pooling_layer,
            nn.Flatten(start_dim, end_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
