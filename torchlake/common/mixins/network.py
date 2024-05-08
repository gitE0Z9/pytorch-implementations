import torch


class SeMixin:
    """Squeeze and excitation mixin, cbam work too"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return self.se(y)
