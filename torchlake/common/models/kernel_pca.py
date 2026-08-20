import enum
from math import sqrt
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


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

    return torch.cdist(x.sqrt(), x.sqrt(), p=2) / sqrt(2)


def linear_kernel_transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.T


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
        y = F.normalize(y, 1, 1)

    return torch.cdist(x.sqrt(), y.sqrt(), p=2) / sqrt(2)


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
        enable_sparse_svd: bool = True,
    ):
        super().__init__()
        self.n_components = n_components
        self.enable_sparse_svd = enable_sparse_svd
        self.col_mean: torch.Tensor
        self.global_mean: torch.Tensor
        self.eigen_vectors: torch.Tensor
        self.eigen_values: torch.Tensor
        self.x_fit: torch.Tensor

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

    def fit(
        self,
        x: torch.Tensor,
        is_x_normalized: bool = False,
        show_progress: bool = True,
    ):
        """fit

        Args:
            x (torch.Tensor): input. shape is (n1, d1)
            is_x_normalized (bool, optional): is x normalized. Defaults to False.
            show_progress (bool, optional): show progress bar. Defaults to True.
        """
        if not is_x_normalized:
            x = F.normalize(x, dim=1, p=1)

        if show_progress:
            progress = tqdm(total=3)

        print("1. kernel computation")
        # TODO: Nyström approximation
        K = self.kernel(x, **self.kernel_params)
        progress.update(1)

        print("2. centering")
        row_mean = K.mean(1, keepdim=True)
        col_mean = K.mean(0, keepdim=True)
        global_mean = col_mean.mean()
        K_centered = K - row_mean - col_mean + global_mean
        # stored for inference
        self.register_buffer("col_mean", col_mean)
        self.register_buffer("global_mean", global_mean)
        progress.update(1)

        print("3. svd")

        if self.enable_sparse_svd:
            # https://docs.pytorch.org/docs/2.13/generated/torch.svd_lowrank.html
            # oversample for quality
            sample_size = x.size(0)
            q = min(self.n_components + 10, sample_size)
            eigvecs, eigvals, _ = torch.svd_lowrank(K_centered, q=q, niter=3)
        else:
            # https://docs.pytorch.org/docs/2.13/generated/torch.linalg.eigh.html
            # returned in ascending order
            eigvals, eigvecs = torch.linalg.eigh(K_centered)
            eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)

        progress.update(1)

        self.register_buffer("eigen_vectors", eigvecs[:, : self.n_components])
        self.register_buffer("eigen_values", eigvals[: self.n_components])
        self.register_buffer("x_fit", x)

    def transform(self, x: torch.Tensor, is_x_normalized: bool = False) -> torch.Tensor:
        """transform

        Args:
            x (torch.Tensor): input. shape is (n2, d1)
            is_x_normalized (bool, optional): is x normalized. Defaults to False.

        Returns:
            torch.Tensor: output. shape is (n2, n_components)
        """
        if not is_x_normalized:
            x = F.normalize(x, dim=1, p=1)

        K_new: torch.Tensor = self.kernel_transform(x, self.x_fit, **self.kernel_params)

        K_new_centered = (
            K_new - K_new.mean(dim=1, keepdim=True) - self.col_mean + self.global_mean
        )

        return K_new_centered @ self.eigen_vectors * self.eigen_values.sqrt()
