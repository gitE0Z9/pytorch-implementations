import torch
from torch import nn

from ..constants import IMAGENET_MEAN, IMAGENET_STD


class ImageNetNormalization(nn.Module):
    def __init__(
        self,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ):
        super(ImageNetNormalization, self).__init__()
        ## C,1,1 shape for broadcasting
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, img: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        original_shape = img.size()

        if img.dim() == 3:
            img = img.unsqueeze(0)

        if not reverse:
            normalized = (img - self.mean.to(img.device)) / self.std.to(img.device)
        else:
            normalized = img * self.std.to(img.device) + self.mean.to(img.device)

        return normalized.reshape(*original_shape)
