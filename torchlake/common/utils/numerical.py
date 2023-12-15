import torch


def safe_sqrt(x: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    return x.pow(2).add(epsilon).sqrt()


def safe_sd(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    return x.var(dim, keepdim=keepdim).add(epsilon).sqrt()
