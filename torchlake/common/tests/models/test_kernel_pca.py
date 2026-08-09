import torch
import pytest

from ...models import KernelPCA
from ...models.kernel_pca import KernelEnum

BATCH_SIZE = 8
VOCAB_SIZE = 100
SEQ_LEN = 10
LATENT_DIM = 2


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
        x = torch.randn(BATCH_SIZE, SEQ_LEN)
        kernel_params = {}

        if kernel == KernelEnum.HELLINGER:
            x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).float()
            kernel_params["is_normalized"] = False

        model = KernelPCA(LATENT_DIM, kernel, kernel_params)

        model.fit(x)

        assert model.eigenvalues.shape == torch.Size((LATENT_DIM,))
        assert model.eigenvectors.shape == torch.Size((BATCH_SIZE, LATENT_DIM))
