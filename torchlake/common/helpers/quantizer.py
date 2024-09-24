import torch
from torch import nn

from ..models.kmeans import KMeans


def get_compression_ratio(
    x: torch.Tensor,
    codebook: torch.Tensor,
    indices: torch.Tensor,
) -> float:
    return (codebook.nbytes + indices.nbytes) / x.nbytes


class KMeansQuantization(nn.Module):
    def __init__(
        self,
        k: int,
        codebook_dtype: torch.dtype = torch.float32,
    ):
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
        self.codebook = []

    def _guess_indices_dtype(self) -> torch.dtype:
        if self.k < 2**8:
            return torch.uint8
        elif self.k < 2**15 - 1:
            return torch.int16
        elif self.k < 2**31 - 1:
            return torch.int32
        else:
            return torch.int64

    def quantize(self, vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """quantize vectors into centroid indices

        Args:
            vectors (list[torch.Tensor]): vectors to quantize, shape is (...other shapes, vector_dimension), size of list is batch_size

        Returns:
            list[torch.Tensor]: cluster indices, shape is (...other shapes), size of list is batch_size
        """
        indices = []
        for vector in vectors:
            model = KMeans(self.k)
            index = model.fit(vector).to(self.indices_dtype)

            self.codebook.append(model.centroids.to(self.codebook_dtype))
            indices.append(index)

        return indices

    def reconstruct(self, indices: torch.Tensor, codebook_id: int = 0) -> torch.Tensor:
        """reconstruct vectors from centroid indices

        Args:
            codebook_id
            indices (torch.Tensor): centroids indices of quantized vectors, shape is (...other shapes).
            codebook_id (int, optional): which codebook to use, Defaults to 0.

        Returns:
            torch.Tensor: reconstructed tensor, shape is (...other shapes, vector_dimension)
        """
        return self.codebook[codebook_id][indices]


class ProductQuantization(nn.Module):
    def __init__(
        self,
        k: int,
        b: int = 8,
        codebook_dtype: torch.dtype = torch.float32,
    ):
        """Product quantization

        Args:
            vector_dim (int): dimension of vector
            k (int): number of subquantizers
            b (int, optional): bits to store centroids. Defaults to 8.
            codebook_dtype (torch.dtype): data type of codebook tensor, could be used for uint8, float16, float32 quantization
        """
        super(ProductQuantization, self).__init__()
        assert k > 0, "subquantizers should be larger than 0"
        assert b > 0, "quantized bits should be longer than 0"

        self.k = k
        self.b = b
        self.codebook_dtype = codebook_dtype
        self.indices_dtype = self._guess_indices_dtype()
        self.codebook = None

    def _guess_indices_dtype(self) -> torch.dtype:
        if self.k < 2**8:
            return torch.uint8
        elif self.k < 2**15 - 1:
            return torch.int16
        elif self.k < 2**31 - 1:
            return torch.int32
        else:
            return torch.int64

    def quantize(self, vectors: torch.Tensor) -> torch.Tensor:
        """quantize vectors into centroid indices

        Args:
            vectors (torch.Tensor): vectors to quantize, shape is (number_vectors, vector_dimension)

        Returns:
            torch.Tensor: centroid indices, shape is (number_vectors, k)
        """
        # chunk into subvectors
        _, embed_dim = vectors.shape
        assert (
            embed_dim % self.k == 0
        ), "number of subquantizers cannot divide vector dimensions"

        vocab_size, embed_dim = vectors.shape
        sub_embed_dim = embed_dim // self.k
        vectors = vectors.view(vocab_size, self.k, 1, sub_embed_dim)

        # each k subvectors has b centroids, each centroid is (embed_dim // k) dimensions
        # randn is random N(0, 1)
        self.codebook = torch.randn(self.k, self.b, sub_embed_dim)

        # find cloest centroid for each k subvectos
        # vocab_size, k, 1, b
        dists = torch.cdist(vectors, self.codebook)

        # compress codebook
        self.codebook = self.codebook.to(self.codebook_dtype)

        # vocab_size, k
        return dists.argmin(-1).squeeze_(-1).to(self.indices_dtype)

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
