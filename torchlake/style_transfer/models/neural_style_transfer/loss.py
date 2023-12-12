import torch
import torch.nn.functional as F
from torch import nn


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_channel, h, w = x.size()

    # batch_size = 1
    features = x.view(batch_size * num_channel, h * w)

    G = torch.mm(features, features.t())  # compute the gram product

    # normalize the values of the gram matrix
    return G / x.numel()


class NeuralStyleTransferLoss(nn.Module):
    def __init__(
        self,
        content_layer_idx: int,
        content_weight: float,
        style_weight: float,
    ):
        super(NeuralStyleTransferLoss, self).__init__()
        self.content_layer_idx = content_layer_idx
        self.content_weight = content_weight
        self.style_weight = style_weight

    def calc_style_loss(
        self,
        feature: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(gram_matrix(feature), gram_matrix(style).detach())

    def calc_content_loss(
        self,
        feature: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(feature, content.detach())

    def forward(
        self,
        content: torch.Tensor,
        styles: list[torch.Tensor],
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor]:
        content_loss = self.calc_content_loss(features[self.content_layer_idx], content)

        style_loss = 0
        for feature, style in zip(features, styles):
            style_loss += self.calc_style_loss(feature, style)

        total_loss = self.content_weight * content_loss + self.style_weight * style_loss

        return total_loss, content_loss, style_loss
