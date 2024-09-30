import abc
from typing import Callable, get_type_hints
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
        if self.k < 2**8:
            return torch.uint8
        elif self.k < 2**15 - 1:
            return torch.int16
        elif self.k < 2**31 - 1:
            return torch.int32
        else:
            return torch.int64

    @abc.abstractmethod
    def quantize(self, vector: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def reconstruct(self, indices: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class KMeansQuantization(Quantizer):
    def __init__(self, k: int, codebook_dtype: torch.dtype = torch.float32):
        """KMeans quantization

        Args:
            k (int): number of clusters
            codebook_dtype (torch.dtype): data type of codebook tensor, could be used for uint8, float16, float32 quantization
        """
        super(KMeansQuantization, self).__init__()
        assert k > 1, "number of clusters should be larger than 1"
        self.k = k
        self.codebook_dtype = codebook_dtype
        self.indices_dtype = self._guess_indices_dtype()
        self.codebook = None

    def quantize(self, vector: torch.Tensor) -> torch.Tensor:
        """quantize vectors into centroid indices

        Args:
            vectors (list[torch.Tensor]): vectors to quantize, shape is (...other shapes, vector_dimension), size of list is batch_size

        Returns:
            list[torch.Tensor]: cluster indices, shape is (...other shapes), size of list is batch_size
        """
        model = KMeans(self.k)
        index = model.fit(vector).to(self.indices_dtype)

        self.codebook = model.centroids.to(self.codebook_dtype)
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
        subquantizer_constructor: Callable[
            [None], Quantizer
        ] = lambda: KMeansQuantization,
        codebook_dtype: torch.dtype = torch.float32,
    ):
        """Product quantization

        Args:
            k (int): number of subquantizers
            b (int, optional): bits to store centroids. Defaults to 8.
            subquantizer_constructor (Callable[[None], Quantizer], optional): subquantizer function. Defaults to lambda:KMeansQuantization.
            codebook_dtype (torch.dtype): data type of codebook tensor, could be used for uint8, float16, float32 quantization
        """
        super(ProductQuantization, self).__init__()
        assert k > 0, "subquantizers should be larger than 0"
        assert b > 0, "quantized bits should be longer than 0"

        self.k = k
        self.b = b
        self.subquantizers = self.build_subquantizer(
            subquantizer_constructor, codebook_dtype
        )

    @property
    def codebook(self) -> torch.Tensor:
        return torch.stack(
            [subquantizer.codebook for subquantizer in self.subquantizers]
        )

    def build_subquantizer(
        self,
        subquantizer_constructor: Callable[
            [None], Quantizer
        ] = lambda: KMeansQuantization,
        codebook_dtype: torch.dtype = torch.float32,
    ) -> Quantizer:
        subquantizer_class = subquantizer_constructor()

        if subquantizer_class == KMeansQuantization:
            return [
                subquantizer_constructor()(self.b, codebook_dtype=codebook_dtype)
                for _ in range(self.k)
            ]
        else:
            raise NotImplementedError("Welcome to implment other subquantizer")

    def quantize(self, vector: torch.Tensor) -> torch.Tensor:
        """quantize vectors into centroid indices

        Args:
            vectors (torch.Tensor): vectors to quantize, shape is (number_vectors, vector_dimension)

        Returns:
            torch.Tensor: centroid indices, shape is (number_vectors, k)
        """
        # chunk into subvectors
        _, embed_dim = vector.shape
        assert (
            embed_dim % self.k == 0
        ), "number of subquantizers cannot divide vector dimensions"

        # k x vocab_size, sub_embed_dim
        vectors = vector.chunk(self.k, -1)

        # total k subvectors, each has b centroids, each centroid is (embed_dim // k) dimensions
        # vocab_size, k
        return torch.stack(
            [
                subquantizer.quantize(vector)
                for vector, subquantizer in zip(vectors, self.subquantizers)
            ],
            -1,
        )

    def reconstruct(self, indices: torch.Tensor) -> torch.Tensor:
        """quantize vectors from centroid indices

        Args:
            indices (torch.Tensor): centroids indices of quantized vectors, shape is (number_vectors, k)

        Returns:
            torch.Tensor: reconstructed tensor
        """
        num_vectors, k = indices.shape
        # vocab_size, embed_dim
        return self.codebook[torch.arange(k), indices].view(num_vectors, -1)
