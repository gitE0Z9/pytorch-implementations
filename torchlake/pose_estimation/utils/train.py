import torch
from torch.distributions import MultivariateNormal
from torchlake.common.utils.numerical import generate_grid


def build_heatmap(
    x: torch.Tensor,
    spatial_shape: tuple[int],
    sigma: float = 1,
) -> torch.Tensor:
    """build guassian heatmap for keypoints

    Args:
        x (torch.Tensor): keypoints, in shape (batch, num_joint, dimension)
        spatial_shape (tuple[int]): size of heatmap, in shape (height, width)
        sigma (float, optional): standard deviation of guassian distribution. Defaults to 1.

    Returns:
        torch.Tensor: heatmap, in shape (batch, num_joint, height, width)
    """
    # check dimension first
    dim = len(spatial_shape)
    msg = "spatial shape should have same dimension as keypoints"
    assert dim == x.size(2), msg

    # declare
    device = x.device
    keypoint_shape = x.shape[:-1]
    mu = torch.zeros(dim).to(device)
    sd = sigma * torch.eye(dim).to(device)
    dist = MultivariateNormal(mu, sd)

    # D x (shape)
    grids = generate_grid(*spatial_shape, indexing="ij")
    # D, *shape
    grids = torch.stack(grids)

    # b*c, D, 1, 1
    points = x.view(-1, dim)[:, :, None, None]
    # b*c, D, N => b*c, N, D => b*c*N, D
    grids = (
        (grids[None, ...] - points)
        .reshape(keypoint_shape.numel(), dim, -1)
        .mT.reshape(-1, dim)
    )

    return dist.log_prob(grids).view(*keypoint_shape, *spatial_shape)
