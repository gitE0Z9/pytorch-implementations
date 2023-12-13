import torch
import torch.nn.functional as F
from torch import nn


def get_multiscale_sizes(
    h: int,
    w: int,
    max_scale: float = 0.5,
    min_size: int = 64,
) -> list[int]:
    max_h, max_w = int(h * max_scale), int(w * max_scale)
    hw_list = [[max_h, max_w]]
    while 1:
        new_h, new_w = max_h // 2, max_w // 2
        if new_h >= min_size or new_w >= min_size:
            hw_list.append([new_h, new_w])
            max_h, max_w = new_h, new_w
        else:
            break
    return hw_list


class MrfLoss(nn.Module):
    def __init__(
        self,
        content_weight: float,
        style_weight: float,
        smoothness: float = 1,
    ):
        super(MrfLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.smoothness = smoothness
        self.unfold = nn.Unfold(3)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.unfold(x).squeeze()

    def find_nn(
        self,
        style_patch: torch.Tensor,
        input_patch: torch.Tensor,
    ) -> torch.Tensor:
        # patch norm
        # C*3*3, (H-2)*(W-2) => pixel_num, patch_num
        input_norm = input_patch / input_patch.norm(p=2, dim=0, keepdim=True)
        style_norm = style_patch / style_patch.norm(p=2, dim=0, keepdim=True)

        # patch x patch
        similarity_matrix = input_norm.t() @ style_norm
        # for input patch, find which style patch is the closet
        nearest_neighbours = similarity_matrix.argmax(dim=1)

        return nearest_neighbours

    def calc_style_loss(
        self,
        style_feature: torch.Tensor,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        style_patch = self.patchify(style_feature)
        input_patch = self.patchify(input_feature)

        knn = self.find_nn(style_patch, input_patch)

        # only style and output compared, no semantic map
        return F.mse_loss(
            input_patch[: (style_feature.size(1) - 3) * 3 * 3],
            style_patch[: (style_feature.size(1) - 3) * 3 * 3, knn],
            reduction="sum",
        )

    def calc_total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        # only output computed, no semantic map
        # tv is calculated loss to right and down
        down_side_loss = F.mse_loss(
            x[:, (x.shape[1] - 3), 1:, :-1],
            x[:, (x.shape[1] - 3), :-1, :-1],
            reduction="sum",
        )
        right_side_loss = F.mse_loss(
            x[:, (x.shape[1] - 3), :-1, 1:],
            x[:, (x.shape[1] - 3), :-1, :-1],
            reduction="sum",
        )
        return right_side_loss + down_side_loss

    def forward(
        self,
        style_features: list[torch.Tensor],
        input_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor]:
        content_loss = 0
        style_loss = 0
        tv_loss = 0

        for style_feature, input_feature in zip(style_features, input_features):
            style_loss += self.calc_style_loss(style_feature, input_feature)
        #         for i in imap: tv_loss += self.total_variation_loss(i)

        total_loss = (
            self.content_weight * content_loss
            + self.style_weight * style_loss
            + self.smoothness * tv_loss
        )

        return total_loss, content_loss, style_loss, tv_loss
