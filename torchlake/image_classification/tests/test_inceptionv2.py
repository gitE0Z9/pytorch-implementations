import pytest
import torch

from ..models.inceptionv2.model import InceptionBN
from ..models.inceptionv2.network import InceptionBlockV2

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
OUTPUT_SIZE = 10


class TestNetwork:
    @pytest.mark.parametrize("stride", (1, 2))
    def test_inception_block_v2_forward_shape(self, stride: int):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionBlockV2(
            HIDDEN_DIM,
            (
                HIDDEN_DIM,
                (HIDDEN_DIM, HIDDEN_DIM),
                (HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),
                HIDDEN_DIM,
            ),
            stride=stride,
        )

        y = m(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 4 * HIDDEN_DIM, IMAGE_SIZE // stride, IMAGE_SIZE // stride)
        )


class TestModel:
    @pytest.mark.parametrize("is_training", (True, False))
    def test_inception_bn_forward_shape(self, is_training: bool):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionBN(INPUT_CHANNEL, OUTPUT_SIZE)

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
