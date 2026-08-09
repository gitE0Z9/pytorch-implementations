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

    return torch.cdist(x.sqrt(), x, p=2) / sqrt(2)


def linear_kernel_transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y


def rbf_kernel_transform(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    z = torch.cdist(x, y, p=2)
    return torch.exp(-gamma * z**2)


def hellinger_kernel_transform(
    x: torch.Tensor,
    y: torch.Tensor,
    is_normalized: bool = True,
) -> torch.Tensor:
    if not is_normalized:
        x = F.normalize(x, 1, 1)

    return torch.cdist(x.sqrt(), y, p=2) / sqrt(2)


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    row_mean = K.mean(1, keepdim=True)
    col_mean = K.mean(0, keepdim=True)
    total_mean = row_mean.mean()
    K_centered = K - row_mean - col_mean + total_mean
    return K_centered


KernelFuncPair = tuple[
    Callable[[torch.Tensor], torch.Tensor],  # fit
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],  # transform
]


class KernelPCA(nn.Module):
    def __init__(
        self,
        n_components: int,
        kernel: str | KernelEnum | KernelFuncPair,
        kernel_params: dict = {},
    ):
        super().__init__()
        self.n_components = n_components

        if isinstance(kernel, str) | isinstance(kernel, KernelEnum):
            self.kernel, self.kernel_transform = self.kernel_mapping[KernelEnum(kernel)]
        else:
            assert (
                len(kernel) == 2
            ), "should provide the kernel function for fit and transform"
            self.kernel, self.kernel_transform = kernel

        self.kernel_params = kernel_params

    @property
    def kernel_mapping(self):
        return {
            KernelEnum.LINEAR: (linear_kernel, linear_kernel_transform),
            KernelEnum.RBF: (rbf_kernel, rbf_kernel_transform),
            KernelEnum.HELLINGER: (hellinger_kernel, hellinger_kernel_transform),
        }

    def fit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        print("1. kernel computation")
        K = self.kernel(x, **self.kernel_params)

        print("2. centering")
        K_centered = center_kernel(K)

        print("3. svd")
        # eigvals, eigvecs = torch.linalg.eigh(K_centered)
        eigvecs, eigvals, _ = torch.linalg.svd(K_centered)

        # eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)

        self.eigenvectors: torch.Tensor = eigvecs[:, : self.n_components]
        self.eigenvalues: torch.Tensor = eigvals[: self.n_components]
        self.x_fit = x

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        K_new: torch.Tensor = self.kernel_transform(x, self.x_fit, **self.kernel_params)

        K_new_centered = (
            K_new
            - K_new.mean(dim=1, keepdim=True)
            - self.x_fit.mean(dim=0)
            + self.x_fit.mean()
        )

        return K_new_centered @ self.eigenvectors * self.eigenvalues.sqrt()
