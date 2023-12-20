import torch


def generate_standard_normal(*size: torch.Size) -> torch.Tensor:
    return torch.empty(size).normal_(0, 1)
