import torch
from math import prod


def eye_matrix(rank: int) -> torch.Tensor:
    indices = torch.arange(rank)
    return torch.sparse_coo_tensor(indices.repeat(2, 1), torch.ones_like(indices))


def ones_tensor(indices: torch.Tensor, size: torch.Size | None = None) -> torch.Tensor:
    assert indices.size(0) == 2, "indices shape is (2, edge)"
    assert indices.size(1) > 0, "indices shape is (2, edge)"

    kwargs = {}
    if size is not None:
        kwargs["size"] = size

    return torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1)).to(indices.device),
        **kwargs,
    )


def get_sparsity(x: torch.Tensor) -> float:
    return 1 - x._nnz() / prod(x.size())
