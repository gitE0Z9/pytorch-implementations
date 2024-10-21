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
    return x.abs().add(epsilon).sqrt()


def safe_std(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    return x.var(dim, keepdim=keepdim).add(epsilon).sqrt()


def log_sum_exp(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """safe log softmax
    compute normalization constant of partition function

    Args:
        x (torch.Tensor): input tensor , shape is (X1, ..... XN)
        keepdim (bool, optional): keep the axis of the reduced dimension, Defaults to False.

    Returns:
        torch.Tensor: output tensor, shape is (X1, ..... XN-1)
    """
    # max_score = x.max(dim, keepdim=True).values
    # return max_score + torch.exp(x - max_score).sum(dim, keepdim=True).log()

    y = x - x.log_softmax(-1)

    if keepdim:
        return y[..., 0:1]
    else:
        return y[..., 0]


def receptive_field(k, l):
    if l == 0:
        return 1
    if l == 1:
        return k
    return (k - 1) ** (l - 1) + receptive_field(k, l - 1)


def generate_grid(grid_x: int, grid_y: int) -> torch.Tensor:
    x_offset, y_offset = torch.meshgrid(
        torch.arange(grid_x),
        torch.arange(grid_y),
        indexing="xy",
    )

    return x_offset, y_offset


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, std: float) -> torch.Tensor:
    z = torch.cdist(x, y) / std
    return torch.exp(-(z**2) / 2)
