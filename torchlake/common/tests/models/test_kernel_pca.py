import torch
import pytest

from ...models import KernelPCA
from ...models.kernel_pca import KernelEnum


class TestKernelPCA:
    @pytest.mark.parametrize(
        "name,kernel",
        [
            (
                "linear_kernel",
                KernelEnum.LINEAR,
            ),
            (
                "rbf_kernel",
                KernelEnum.RBF,
            ),
            (
                "helligner_kernel",
                KernelEnum.HELLINGER,
            ),
        ],
    )
    def test_output_shape(self, name: str, kernel: str):
        x = torch.randn(8, 10)
        kernel_params = {}

        if kernel == KernelEnum.HELLINGER:
            x = torch.randint(0, 100, (8, 10)).float()
            kernel_params["is_normalized"] = False

        model = KernelPCA(2, kernel, kernel_params)

        model.fit(x)

        assert model.eigenvalues.shape == torch.Size((2,))
        assert model.eigenvectors.shape == torch.Size((8, 2))
