from unittest import TestCase

import pytest
import torch
from parameterized import parameterized

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
        # self.i = torch.randint(0, self.k, (224, 224), dtype=torch.uint8)

    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    def test_quantize(self, codebook_dtype: torch.dtype):
        self.setUp()
        model = KMeansQuantization(self.k, codebook_dtype=codebook_dtype)
        i = model.quantize([self.x.float()])

        assert i[0].shape == torch.Size((224, 224))
        assert i[0].dtype == torch.uint8
        assert model.codebook[0].shape == torch.Size((self.k, 3))
        assert model.codebook[0].dtype == codebook_dtype

    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    def test_reconstruct(self, codebook_dtype: torch.dtype):
        self.setUp()
        model = KMeansQuantization(self.k, codebook_dtype=codebook_dtype)
        i = model.quantize([self.x.float()])
        x_prime = model.reconstruct(i[0].long())

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
    def test_quantize(self, k: int, b: int, codebook_dtype: torch.dtype):
        self.setUp()
        model = ProductQuantization(k, b, codebook_dtype=codebook_dtype)
        i = model.quantize(self.x)

        assert i.shape == torch.Size((self.n, k))
        assert i.dtype == torch.uint8
        assert model.codebook.shape == torch.Size((k, b, self.d // k))
        assert model.codebook.dtype == codebook_dtype

    @pytest.mark.parametrize("k", [5, 10])
    @pytest.mark.parametrize("b", [8])
    @pytest.mark.parametrize(
        "codebook_dtype", [torch.uint8, torch.float16, torch.float32]
    )
    def test_reconstruct(self, k: int, b: int, codebook_dtype: torch.dtype):
        self.setUp()
        model = ProductQuantization(k, b, codebook_dtype=codebook_dtype)
        i = model.quantize(self.x)
        x_prime = model.reconstruct(i.long())

        assert x_prime.shape == torch.Size((self.n, self.d))
        assert x_prime.dtype == codebook_dtype
