import torch


class SEMixin:
    """Squeeze and excitation mixin, cbam work too"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.se is not None:
            return self.se(y)

        return y
