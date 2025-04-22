import math
from typing import Literal

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


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


def receptive_field(k: int, l: int) -> int:
    """calculate size of receptive field of convolution filter

    Args:
        k (int): kernel size
        l (int): layer 1-index

    Returns:
        int: receptive field of convolution filter
    """
    if l == 0:
        return 1
    if l == 1:
        return k
    return (k - 1) ** (l - 1) + receptive_field(k, l - 1)


def generate_grid(
    *shapes: tuple[int],
    centered: bool = False,
    indexing: Literal["xy", "ij"] = "xy",
) -> tuple[torch.Tensor]:
    """generate

    Args:
        shapes (tuple[int]): grid shape
        centered (bool, optional): use manhattan distance to the center of the grid. Defaults to False.
        indexing (Literal["xy", "ij"], optional): xy will flip, ij will preserve the order. Defaults to "xy".

    Returns:
        tuple[torch.Tensor]: grid
    """
    grids = torch.meshgrid(
        [torch.arange(shape) for shape in shapes],
        # xy will switch first 2 dim
        # so (h,w) => x,y
        # (t, h, w) => y,z,x
        indexing=indexing,
    )

    if centered:
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


def build_gaussian_heatmap(
    x: torch.Tensor,
    spatial_shape: tuple[int],
    sigma: float = 1,
    effective_range: int = 3,
    normalized: bool = True,
    amplitude: float = 1,
    truncated: bool = False,
) -> torch.Tensor:
    """build guassian heatmap for keypoints

    Args:
        x (torch.Tensor): keypoints, in shape (batch, num_joint, dimension)
        spatial_shape (tuple[int]): size of heatmap, in shape (height, width)
        sigma (float, optional): standard deviation of guassian distribution. Defaults to 1.
        effective_range (bool, optional): log probability is only effective in the neighborhood within the range. Defaults to 3.
        normalized (bool, optional): normalized gaussian pdf. Defaults to True.
        amplitude (float, optional): multiply heatmap with amplitude. Defaults to 1.
        truncated (bool, optional): use truncated normal distribution, can only be used when effective is True. Defauts to False.

    Returns:
        torch.Tensor: heatmap, in shape (batch, num_joint, height, width)
    """
    # check dimension first
    dim = len(spatial_shape)
    msg = "spatial shape should have same dimension as keypoints"
    assert dim == x.size(2), msg

    # declare
    device = x.device
    # batch, num_joint
    keypoint_shape = x.shape[:-1]
    num_points = math.prod(spatial_shape)

    grids = generate_grid(*spatial_shape, indexing="ij")
    # D, ...spatial
    grids = torch.stack(grids).to(device)

    # b*c, D, 1, 1
    points = x.view(-1, dim)[:, :, None, None]
    # b*c, D, N => b*c, N, D => b*c*N, D
    grids = (
        (grids[None, ...] - points)
        .reshape(keypoint_shape.numel(), dim, -1)
        .mT.reshape(-1, dim)
    )

    # TODO: should upgrade! www
    # torch 2.1.0 fix
    # https://discuss.pytorch.org/t/how-to-use-torch-distributions-multivariate-normal-multivariatenormal-in-multi-gpu-mode/135030/3
    # manually control batch size, so gpu runtime error is not triggered
    n = grids.size(0)
    mu = torch.zeros(n, dim).to(device)
    sd = sigma * torch.eye(dim).unsqueeze(0).repeat(n, *(1,) * dim).to(device)
    dist = MultivariateNormal(mu, sd**2)
    # b*c, N
    hm: torch.Tensor = dist.log_prob(grids).view(-1, num_points)

    if effective_range > 0:
        # simplify computation by 1d
        ndist = Normal(torch.Tensor([0]), torch.Tensor([sigma]))
        threshold: torch.Tensor = ndist.log_prob(torch.Tensor([effective_range])) + (
            dim - 1
        ) * ndist.log_prob(torch.Tensor([0]))
        hm[hm < threshold.item()] = -torch.inf

        if truncated:
            hm -= hm.logsumexp(dim=-1, keepdim=True)
            hm[hm.isnan()] = -torch.inf

    if not normalized:
        hm += dim * (math.log(2 * math.pi) / 2 + math.log(sigma))

    if amplitude != 1:
        hm += math.log(amplitude)

    return hm.view(*keypoint_shape, *spatial_shape)
