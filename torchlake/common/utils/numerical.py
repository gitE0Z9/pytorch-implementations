from typing import Literal
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


def log_sum_exp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """compute log softmax in numerically stable way
    often used to compute normalization constant of partition function

    Args:
        x (torch.Tensor): input tensor , shape is (X1, ..... XN)
        dim (int, optional): reduced dimension. Defaults to -1.
        keepdim (bool, optional): keep the axis of the reduced dimension, Defaults to False.

    Returns:
        torch.Tensor: output tensor, shape is (X1, ..... XN-1)
    """
    # this is max value trick, since it is slower, left here for demonstration purpose only
    # max_score = x.max(dim, keepdim=True).values
    # return max_score + torch.exp(x - max_score).sum(dim, keepdim=True).log()

    y = x - x.log_softmax(dim)

    y = y.narrow(dim, 0, 1)

    return y if keepdim else y.squeeze(dim)


def receptive_field(k, l):
    if l == 0:
        return 1
    if l == 1:
        return k
    return (k - 1) ** (l - 1) + receptive_field(k, l - 1)


def generate_grid(
    *shapes: tuple[int],
    center: bool = False,
    indexing: Literal["xy", "ij"] = "xy",
) -> tuple[torch.Tensor]:
    grids = torch.meshgrid(
        [torch.arange(shape) for shape in shapes],
        # xy will switch first 2 dim
        # so (h,w) => x,y
        # (t, h, w) => y,z,x
        indexing=indexing,
    )

    if center:
        grids = list(grids)
        for i in range(len(shapes)):
            mid = shapes[i] // 2
            if shapes[i] % 2 == 0:
                mid = (2 * mid + 1) / 2

            grids[i] = grids[i] - mid

    return grids


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, std: float) -> torch.Tensor:
    z = torch.cdist(x, y) / std
    return torch.exp(-(z**2) / 2)


def padded_mask(x: torch.Tensor, padding_idx: int) -> torch.Tensor:
    return x.eq(padding_idx)


def causal_mask(*shape: int) -> torch.Tensor:
    return torch.tril(torch.ones(shape))
