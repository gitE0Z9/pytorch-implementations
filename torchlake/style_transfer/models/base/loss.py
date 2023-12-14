import torch


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d = x.shape
    y = x.reshape(a, b, c * d)
    y = torch.bmm(y, y.transpose(1, 2))
    y = y / (b * c * d)

    return y
