import pytest
import torch

from ..helpers.quantizer import (
    KMeansQuantization,
    ProductQuantization,
    get_compression_ratio,
)


@pytest.mark.parametrize(
    "x,codebook,indices,expected",
    [
        [
            torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
            torch.randint(0, 5, (3, 5), dtype=torch.uint8),
            torch.randint(0, 5, (224, 224), dtype=torch.uint8),
            (3 * 5 * 8 * 1 + 224 * 224 * 8 * 1) / (3 * 224 * 224 * 8 * 1),
        ],
        [
            torch.rand(3, 224, 224),
            torch.rand(3, 5),
            torch.randint(0, 5, (224, 224), dtype=torch.uint8),
            (3 * 5 * 8 * 4 + 224 * 224 * 8 * 1) / (3 * 224 * 224 * 8 * 4),
        ],
    ],
)
def test_get_compression_ratio(
    x: torch.Tensor,
    codebook: torch.Tensor,
    indices: torch.Tensor,
    expected: float,
):
    assert get_compression_ratio(x, codebook, indices) == expected


class TestKMeansQuantization:
    def setUp(self) -> None:
        self.k = 5
        self.x = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)

    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    def test_quantize(self, codebook_dtype: torch.dtype):
        self.setUp()
        model = KMeansQuantization(self.k, codebook_dtype=codebook_dtype)
        i = model.quantize(self.x.float())

        assert i.shape == torch.Size((224, 224))
        assert i.dtype == torch.uint8
        assert model.codebook.shape == torch.Size((self.k, 3))
        assert model.codebook.dtype == codebook_dtype

    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    def test_reconstruct(self, codebook_dtype: torch.dtype):
        self.setUp()
        model = KMeansQuantization(self.k, codebook_dtype=codebook_dtype)
        i = model.quantize(self.x.float())
        x_prime = model.reconstruct(i.long())

        assert x_prime.shape == torch.Size((224, 224, 3))
        assert x_prime.dtype == codebook_dtype


class TestProductQuantization:
    def setUp(self) -> None:
        self.n = 10000
        self.d = 300
        self.x = torch.rand(self.n, self.d)

    @pytest.mark.parametrize("k", [5, 10])
    @pytest.mark.parametrize("b", [8])
    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    @pytest.mark.parametrize("normalized", [True, False])
    def test_quantize(
        self,
        k: int,
        b: int,
        codebook_dtype: torch.dtype,
        normalized: bool,
    ):
        self.setUp()
        model = ProductQuantization(
            k,
            b,
            codebook_dtype=codebook_dtype,
            normalized=normalized,
        )
        indices = model.quantize(self.x)
        codebook = model.codebook

        if normalized:
            norm_indices, indices = indices
            assert norm_indices.shape == torch.Size((self.n, k))
            assert norm_indices.dtype == torch.uint8
            norm_codebook, codebook = codebook
            assert norm_codebook.shape == torch.Size((k, 2**b, 1))
            assert norm_codebook.dtype == codebook_dtype

        assert indices.shape == torch.Size((self.n, k))
        assert indices.dtype == torch.uint8
        assert codebook.shape == torch.Size((k, 2**b, self.d // k))
        assert codebook.dtype == codebook_dtype

    @pytest.mark.parametrize("k", [5, 10])
    @pytest.mark.parametrize("b", [8])
    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    @pytest.mark.parametrize("normalized", [True, False])
    def test_reconstruct(
        self,
        k: int,
        b: int,
        codebook_dtype: torch.dtype,
        normalized: bool,
    ):
        self.setUp()
        model = ProductQuantization(
            k,
            b,
            codebook_dtype=codebook_dtype,
            normalized=normalized,
        )
        indices = model.quantize(self.x)
        norm_indices = None
        if normalized:
            norm_indices, indices = indices

        x_prime = model.reconstruct(
            indices.long(),
            norm_indices=norm_indices.long() if norm_indices is not None else None,
        )

        assert x_prime.shape == torch.Size((self.n, self.d))
        assert x_prime.dtype == codebook_dtype
