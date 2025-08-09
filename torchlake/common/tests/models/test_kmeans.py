import pytest
import torch

from ...models.kmeans import KMeans


class TestKMeans:
    @pytest.mark.parametrize("k", [3, 5, 10])
    @pytest.mark.parametrize("init_method", ["random", "kmeans++"])
    def test_output_shape(self, k: int, init_method: str):
        x = torch.randn(100, 10)
        model = KMeans(k, init_method=init_method)
        indices = model.fit(x)

        assert indices.shape == torch.Size((100,))
        assert model.centroids.shape == torch.Size((k, 10))

    @pytest.mark.parametrize("k", [3, 5, 10])
    @pytest.mark.parametrize("init_method", ["random", "kmeans++"])
    def test_transform_shape(self, k: int, init_method: str):
        x = torch.randn(100, 10)
        model = KMeans(k, init_method=init_method)
        indices = model.fit(x)
        y = model.transform(indices)

        assert y.shape == torch.Size((100, 10))
