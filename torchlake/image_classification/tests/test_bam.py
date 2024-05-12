import pytest
import torch

from ..models.bam.model import BamResNet


class TestBamResNet:
    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["18", 18],
            ["34", 34],
            ["50", 50],
            ["101", 101],
            ["152", 152],
        ],
    )
    def test_resnet_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = BamResNet(output_size=5, num_layer=num_layer)
        y = model(x)

        assert y.shape == torch.Size((2, 5))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["18", 18],
            ["34", 34],
            ["50", 50],
            ["101", 101],
            ["152", 152],
        ],
    )
    def test_resnet2_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = BamResNet(output_size=5, num_layer=num_layer, pre_activation=True)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
