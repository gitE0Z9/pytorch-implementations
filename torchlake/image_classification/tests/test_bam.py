import pytest
import torch

from ..models.bam.model import BamResNet
from ..models.bam.network import Bam2d


class TestBam2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 112, 112)

        model = Bam2d(16, 16)

        y = model(x)

        assert y.shape == torch.Size((8, 16, 112, 112))


class TestBamResNet:
    @pytest.mark.parametrize("key", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resnet_forward_shape(self, key: int, pre_activation: bool):
        x = torch.randn(2, 3, 224, 224)
        model = BamResNet(
            output_size=5,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))
