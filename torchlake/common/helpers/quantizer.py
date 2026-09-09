import abc
from typing import Callable
import torch
from torch import nn

from ..models.kmeans import KMeans


def get_compression_ratio(
    x: torch.Tensor,
    codebook: torch.Tensor,
    indices: torch.Tensor,
) -> float:
    return (codebook.nbytes + indices.nbytes) / x.nbytes


class Quantizer(nn.Module, abc.ABC):
    def _guess_indices_dtype(self) -> torch.dtype:
        if self.k <= 2**8:
            return torch.uint8
        elif self.k <= 2**16:
            return torch.uint16
        elif self.k <= 2**32:
            return torch.uint32
        else:
            return torch.uint64

    @abc.abstractmethod
    def quantize(self, vector: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def reconstruct(self, indices: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class KMeansQuantization(Quantizer):
    def __init__(
        self,
        k: int,
        codebook_dtype: torch.dtype = torch.float32,
        indices_dtype: torch.dtype | None = None,
    ):
        """KMeans quantization

        Args:
            k (int): number of clusters
            codebook_dtype (torch.dtype): data type of codebook tensor
            indices_dtype (torch.dtype, optional): data type of indices tensor. Default to None.
        """
        super().__init__()
        assert k > 1, "number of clusters should be larger than 1"
        self.k = k
        self.codebook_dtype = codebook_dtype
        self.indices_dtype = indices_dtype or self._guess_indices_dtype()
        self.codebook = None

    def quantize(self, vector: torch.Tensor) -> torch.Tensor:
        """quantize vectors into centroid indices

        Args:
            vectors (list[torch.Tensor]): vectors to quantize, shape is (...other shapes, vector_dimension), size of list is batch_size

        Returns:
            list[torch.Tensor]: cluster indices, shape is (...other shapes), size of list is batch_size
        """
        model = KMeans(self.k)
        # ...
        index = model.fit(vector).to(self.indices_dtype)

        # k, d
        self.codebook = nn.Parameter(
            model.centroids.to(self.codebook_dtype),
            requires_grad=False,
        )

        # ...
        return index

    def reconstruct(self, indices: torch.Tensor) -> torch.Tensor:
        """reconstruct vectors from centroid indices

        Args:
            codebook_id
            indices (torch.Tensor): centroids indices of quantized vectors, shape is (...other shapes).

        Returns:
            torch.Tensor: reconstructed tensor, shape is (...other shapes, vector_dimension)
        """
        return self.codebook[indices]


class ProductQuantization(Quantizer):
    def __init__(
        self,
        k: int,
        b: int = 8,
        subquantizer_constructor: Quantizer = KMeansQuantization,
        codebook_dtype: torch.dtype = torch.float32,
        indices_dtype: torch.dtype | None = None,
        normalized: bool = False,
    ):
        """Product quantization

        Args:
            k (int): number of subquantizers
            b (int, optional): bits to store centroids. Defaults to 8.
            subquantizer_constructor (Quantizer, optional): subquantizer function. Defaults to KMeansQuantization.
            codebook_dtype (torch.dtype): data type of codebook tensor.
            indices_dtype (torch.dtype, optional): data type of indices tensor. Default to None.
            normalized (bool, optional): enable normalized product quantization. Defaults to False.
        """
        super().__init__()
        assert k > 0, "subquantizers should be larger than 0"
        assert b > 0, "quantized bits should be longer than 0"

        self.k = k
        self.b = b
        self.normalized = normalized
        self.codebook_dtype = codebook_dtype
        self.indices_dtype = indices_dtype or self._guess_indices_dtype()
        self.subquantizers = self.build_subquantizer(subquantizer_constructor)

    def _guess_indices_dtype(self) -> torch.dtype:
        if self.b <= 8:
            return torch.uint8
        elif self.b <= 16:
            return torch.uint16
        elif self.b <= 32:
            return torch.uint32
        else:
            return torch.uint64

    @property
    def codebook(self) -> torch.Tensor:
        if self.normalized:
            return (
                # k, 2**b, 1
                torch.stack(
                    [
                        norm_subquantizer.codebook
                        for norm_subquantizer in self.subquantizers[0]
                    ],
                ),
                # k, 2**b, d//k
                torch.stack(
                    [
                        scaled_subquantizer.codebook
                        for scaled_subquantizer in self.subquantizers[1]
                    ],
                ),
            )
        else:
            # k, 2**b, d//k
            return torch.stack(
                [subquantizer.codebook for subquantizer in self.subquantizers[0]]
            )

    def build_subquantizer(
        self,
        subquantizer_constructor: Quantizer = KMeansQuantization,
    ) -> nn.ModuleList | tuple[nn.ModuleList, nn.ModuleList]:
        if subquantizer_constructor == KMeansQuantization:
            groups = 2 if self.normalized else 1

            return nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            subquantizer_constructor(
                                k=int(2**self.b),
                                codebook_dtype=self.codebook_dtype,
                                indices_dtype=self.indices_dtype,
                            )
                            for _ in range(self.k)
                        ]
                    )
                    for _ in range(groups)
                ]
            )
        else:
            raise NotImplementedError("Welcome to implment other subquantizer")

    def quantize(self, vector: torch.Tensor) -> torch.Tensor:
        """quantize vectors into centroid indices

        Args:
            vectors (torch.Tensor): vectors to quantize, shape is (..., vector_dimension)

        Returns:
            torch.Tensor: centroid indices, shape is (..., k)
        """
        # chunk into subvectors
        embed_dim = vector.size(-1)
        assert (
            embed_dim % self.k == 0
        ), "number of subquantizers cannot divide vector dimensions"

        # k x (..., sub_embed_dim)
        vectors = vector.chunk(self.k, -1)

        if self.normalized:
            # k x (..., 1)
            norms = [vector.sum(-1, keepdim=True) for vector in vectors]
            # k x (..., sub_embed_dim)
            vectors = [vectors[i] / norms[i] for i in range(self.k)]
            return (
                # ..., k
                torch.stack(
                    [
                        subquantizer.quantize(norm)
                        for norm, subquantizer in zip(norms, self.subquantizers[0])
                    ],
                    -1,
                ),
                # ..., k
                torch.stack(
                    [
                        subquantizer.quantize(vector)
                        for vector, subquantizer in zip(vectors, self.subquantizers[1])
                    ],
                    -1,
                ),
            )
        else:
            # total k subvectors, each has 2**b centroids, each centroid has (embed_dim // k) dimensions
            # ..., k
            return torch.stack(
                [
                    subquantizer.quantize(vector)
                    for vector, subquantizer in zip(vectors, self.subquantizers[0])
                ],
                -1,
            )

    def reconstruct(
        self,
        indices: torch.Tensor,
        norm_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """quantize vectors from centroid indices

        Args:
            indices (torch.Tensor): centroids indices of quantized vectors, shape is (..., k)
            norm_indices (torch.Tensor): centroids indices of quantized vector norms, shape is (..., k)

        Returns:
            torch.Tensor: reconstructed tensor
        """
        if self.normalized:
            assert norm_indices is not None, "norm indices must be provided"

            # ..., k, 1
            norm = self.codebook[0][torch.arange(self.k), norm_indices]
            # ..., k, d//k
            scaled = self.codebook[1][torch.arange(self.k), indices]

            # ..., k, d//k => ..., d
            return (norm * scaled).view(*indices.shape[:-1], -1)
        else:
            # ..., k, d//k => ..., d
            return self.codebook[torch.arange(self.k), indices].view(
                *indices.shape[:-1], -1
            )
