import torch


class SeMixin:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return self.se(y)
