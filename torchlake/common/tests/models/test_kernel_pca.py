import torch
import pytest

from ...models import KernelPCA
from ...models.kernel_pca import KernelEnum

SAMPLE_SIZE = 32
VOCAB_SIZE = 100
SEQ_LEN = 10
LATENT_DIM = 2
INPUT_CHANNEL = 4


class TestKernelPCA:
    @pytest.mark.parametrize(
        "name,kernel,kernel_params",
        [
            (
                "linear_kernel",
                KernelEnum.LINEAR,
                {},
            ),
            (
                "rbf_kernel",
                KernelEnum.RBF,
                {},
            ),
            (
                "helligner_kernel",
                KernelEnum.HELLINGER,
                {},
            ),
        ],
    )
    def test_fit_output_shape(self, name: str, kernel: str, kernel_params: dict):
        x = torch.rand(SAMPLE_SIZE, INPUT_CHANNEL)

        model = KernelPCA(LATENT_DIM, kernel, kernel_params)

        model.fit(x)

        assert model.col_mean is not None
        assert model.global_mean is not None
        assert model.eigen_vectors is not None
        assert model.eigen_values is not None
        assert model.x_fit is not None

        assert model.col_mean.shape == torch.Size((1, SAMPLE_SIZE))
        assert model.eigen_vectors.shape == torch.Size((SAMPLE_SIZE, LATENT_DIM))
        assert model.eigen_values.shape == torch.Size((LATENT_DIM,))
        assert model.x_fit.shape == torch.Size((SAMPLE_SIZE, INPUT_CHANNEL))

    @pytest.mark.parametrize(
        "name,kernel,kernel_params",
        [
            (
                "linear_kernel",
                KernelEnum.LINEAR,
                {},
            ),
            (
                "rbf_kernel",
                KernelEnum.RBF,
                {},
            ),
            (
                "helligner_kernel",
                KernelEnum.HELLINGER,
                {},
            ),
        ],
    )
    def test_transform_output_shape(self, name: str, kernel: str, kernel_params: dict):
        x = torch.rand(SAMPLE_SIZE, INPUT_CHANNEL)
        y = torch.rand(SAMPLE_SIZE * 2, INPUT_CHANNEL)

        model = KernelPCA(LATENT_DIM, kernel, kernel_params)

        model.fit(x)
        output = model.transform(y)

        assert output.shape == torch.Size((SAMPLE_SIZE * 2, LATENT_DIM))
