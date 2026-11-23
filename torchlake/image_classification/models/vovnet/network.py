import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class OSAModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_channel: int,
        num_branch: int = 5,
    ):
        """One-shot aggregation module [1904.09730]
        a chain of 3x3 convs, the input and every intermediate output are concatenated once at the end,
        then aggregated by a 1x1 conv.

        Args:
            input_channel (int): input channel size.
            hidden_dim (int): output channel size of each 3x3 conv.
            output_channel (int): output channel size of the 1x1 aggregation conv.
            num_branch (int, optional): number of 3x3 convs. Defaults to 5.
        """
        super().__init__()
        self.branches = nn.ModuleList(
            [
                Conv2dNormActivation(
                    input_channel if i == 0 else hidden_dim,
                    hidden_dim,
                    3,
                )
                for i in range(num_branch)
            ]
        )
        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel + num_branch * hidden_dim,
                output_channel,
                1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for branch in self.branches:
            features.append(branch(features[-1]))
        y = torch.cat(features, 1)

        return self.layers(y)
