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
    n = K.shape[0]
    ones = torch.ones((n, n), device=K.device) / n
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    return K_centered


class KernelPCA(nn.Module):
    def __init__(
        self,
        n_components: int,
        kernel: KernelEnum | Callable[[torch.Tensor], torch.Tensor],
        kernel_params: dict = {},
    ):
        self.n_components = n_components
        self.kernel = (
            kernel if isinstance(kernel, Callable) else self.kernel_mapping[kernel]
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

        # Step 1: Compute the kernel matrix
        K = self.kernel(x, **self.kernel_params)

        # Step 2: Center the kernel matrix
        K_centered = center_kernel(K)

        # Step 3: Eigenvalue decomposition
        eigvals, eigvecs = torch.linalg.eigh(K_centered)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)

        # Step 5: Select the top n_components eigenvectors
        self.eigenvectors: torch.Tensor = eigvecs[:, : self.n_components]
        self.eigenvalues: torch.Tensor = eigvals[: self.n_components]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the kernel between the new data and the fitted data
        K_new: torch.Tensor = self.kernel(x, self.x_fit, **self.kernel_params)

        # Center the new kernel matrix
        K_new_centered = (
            K_new
            - K_new.mean(dim=1, keepdim=True)
            - self.x_fit.mean(dim=0)
            + self.x_fit.mean()
        )

        # Project the new data into the PCA space
        return K_new_centered @ self.eigenvectors * self.eigenvalues.sqrt()
