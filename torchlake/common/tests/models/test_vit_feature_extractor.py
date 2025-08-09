import pytest
import torch

from ...models.vit_feature_extractor import ViTFeatureExtractor


class TestViTFeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    @pytest.mark.parametrize(
        "network_name,target_layer_names,seq_len",
        [
            ("b16", list(range(12)) + ["output"], 196),
            ("b32", list(range(12)) + ["output"], 49),
            ("l16", list(range(24)) + ["output"], 196),
            ("l32", list(range(24)) + ["output"], 49),
        ],
    )
    def test_output_shape(
        self,
        network_name: str,
        target_layer_names: list[int],
        seq_len: int,
    ):
        self.setUp()
        model = ViTFeatureExtractor(network_name)
        y = model.forward(self.x, target_layer_names)

        for dim in model.feature_dims:
            assert y.pop(0).shape == torch.Size((1, seq_len, dim))

        assert y.pop(0).shape == torch.Size((1, seq_len, dim))
