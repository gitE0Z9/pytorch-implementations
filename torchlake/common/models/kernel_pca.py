import enum
from math import sqrt
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class KernelEnum(enum.Enum):
    LINEAR = "linear"
    RBF = "rbf"
    HELLINGER = "hellinger"


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T


def rbf_kernel(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    y = torch.cdist(x, x, p=2)
    return torch.exp(-gamma * y**2)


def hellinger_kernel(x: torch.Tensor, is_normalized: bool = True) -> torch.Tensor:
    if not is_normalized:
        x = F.normalize(x, 1, 1)
    x = x.sqrt()
    return torch.cdist(x, x, p=2) / sqrt(2)


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    N = K.shape[0]
    row_mean = K.mean(1, keepdim=True) / N
    col_mean = K.mean(0, keepdim=True) / N
    total_mean = row_mean.mean()
    K_centered = K - row_mean - col_mean + total_mean
    return K_centered


class KernelPCA(nn.Module):
    def __init__(
        self,
        n_components: int,
        kernel: str | KernelEnum | Callable[[torch.Tensor], torch.Tensor],
        kernel_params: dict = {},
    ):
        super(KernelPCA, self).__init__()
        self.n_components = n_components
        self.kernel = (
            kernel
            if isinstance(kernel, Callable)
            else self.kernel_mapping[KernelEnum(kernel)]
        )
        self.kernel_params = kernel_params

    @property
    def kernel_mapping(self):
        return {
            KernelEnum.LINEAR: linear_kernel,
            KernelEnum.RBF: rbf_kernel,
            KernelEnum.HELLINGER: hellinger_kernel,
        }

    def fit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.x_fit = x

        K = self.kernel(x, **self.kernel_params)

        K_centered = center_kernel(K)

        # eigvals, eigvecs = torch.linalg.eigh(K_centered)
        eigvecs, eigvals, _ = torch.linalg.svd(K_centered)

        # eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)

        self.eigenvectors: torch.Tensor = eigvecs[:, : self.n_components]
        self.eigenvalues: torch.Tensor = eigvals[: self.n_components]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        K_new: torch.Tensor = self.kernel(x, self.x_fit, **self.kernel_params)

        K_new_centered = (
            K_new
            - K_new.mean(dim=1, keepdim=True)
            - self.x_fit.mean(dim=0)
            + self.x_fit.mean()
        )

        return K_new_centered @ self.eigenvectors * self.eigenvalues.sqrt()
