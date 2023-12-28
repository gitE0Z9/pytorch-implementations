import torch


def safe_negative_factorial(
    x: torch.Tensor,
    negative_factorial: float,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    assert negative_factorial < 0, "negative_factorial must be negative"
    assert negative_factorial > -1, "negative_factorial must be larger than -1"

    factorial = abs(negative_factorial)

    return x.pow(1 / factorial).add(epsilon).pow(factorial)


def safe_sqrt(x: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    return x.pow(2).add(epsilon).sqrt()


def safe_std(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    return x.var(dim, keepdim=keepdim).add(epsilon).sqrt()


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """safe log softmax"""
    max_score, _ = x.max(-1, keepdim=True)  # B, L, 1
    return max_score + torch.exp(x - max_score).sum(-1, keepdim=True).log()
