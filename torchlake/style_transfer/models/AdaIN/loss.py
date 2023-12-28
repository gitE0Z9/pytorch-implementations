import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.utils.numerical import safe_std


class AdaInLoss(nn.Module):
    def __init__(self, lambda_coef: float = 1e5):
        super(AdaInLoss, self).__init__()
        self.lambda_coef = lambda_coef

    def calc_content_loss(
        self,
        generated_feature: torch.Tensor,
        normalized_content: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(generated_feature, normalized_content.detach())

    def calc_style_loss(
        self,
        generated_features: list[torch.Tensor],
        style_features: list[torch.Tensor],
    ) -> torch.Tensor:
        moment_1st_loss = 0
        moment_2nd_loss = 0

        for generated_feature, style_feature in zip(generated_features, style_features):
            moment_1st_loss += F.mse_loss(
                generated_feature.mean((2, 3)),
                style_feature.mean((2, 3)).detach(),
            )
            moment_2nd_loss += F.mse_loss(
                safe_std(generated_feature, (2, 3)),
                safe_std(style_feature, (2, 3)).detach(),
            )

        return moment_1st_loss + moment_2nd_loss

    def forward(
        self,
        generated_features: list[torch.Tensor],
        normalized_content: torch.Tensor,
        style_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor]:
        content_loss = self.calc_content_loss(
            generated_features[-1],
            normalized_content,
        )
        style_loss = self.calc_style_loss(generated_features, style_features)

        return content_loss + self.lambda_coef * style_loss, content_loss, style_loss
