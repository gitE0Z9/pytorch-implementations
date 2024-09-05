from typing import Literal

import torch
from torch import nn


class FlattenFeature(nn.Module):

    def __init__(
        self,
        reduction: Literal["mean"] | Literal["max"] | None = "mean",
        dimension: Literal["1d"] | Literal["2d"] | Literal["3d"] = "2d",
    ):
        """_summary_

        Args:
            reduction (Literal[mean] | Literal[max] | None, optional): redution mode. Defaults to "mean".
            dimension (Literal["1d"] | Literal["2d"] | Literal["3d"], optional): 1d, 2d or 3d. Defaults to "2d".
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
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
