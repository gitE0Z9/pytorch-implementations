import pytest
import torch

from ..models.inception.model import GoogLeNet
from ..models.inception.network import AuxiliaryClassifier, InceptionBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
OUTPUT_SIZE = 10


class TestNetwork:
    def test_inception_block_forward_shape(self):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionBlock(
            HIDDEN_DIM,
            (
                HIDDEN_DIM,
                (HIDDEN_DIM, HIDDEN_DIM),
                (HIDDEN_DIM, HIDDEN_DIM),
                HIDDEN_DIM,
            ),
        )

        y = m(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 4 * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        )

    def test_auxiliary_classifier_forward_shape(self):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        m = AuxiliaryClassifier(
            HIDDEN_DIM,
            OUTPUT_SIZE,
            hidden_dims=(HIDDEN_DIM, HIDDEN_DIM),
        )

        y = m(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))


class TestModel:
    @pytest.mark.parametrize("is_training", (True, False))
    @pytest.mark.parametrize("legacy", (True, False))
    def test_googlenet_forward_shape(self, is_training: bool, legacy: bool):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = GoogLeNet(INPUT_CHANNEL, OUTPUT_SIZE, legacy=legacy)

        if is_training:
            m.train()
        else:
            m.eval()

        y = m(x)

        if is_training:
            for ele in y:
                assert ele.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
        else:
            assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
