import pytest
import torch

from ..models.inceptionv4.model import InceptionV4

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 299
NUM_CLASS = 10


class TestModel:
    def test_inception_v4_forward_shape(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionV4(INPUT_CHANNEL, NUM_CLASS)

        y = m(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS))
