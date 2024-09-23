import torch


def get_tensor_bits(x: torch.Tensor) -> int:
    return x.dtype.itemsize * 8 * x.numel()
