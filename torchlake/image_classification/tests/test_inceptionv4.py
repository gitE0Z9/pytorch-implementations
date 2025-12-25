import torch

from ..models.inceptionv4.model import InceptionV4, InceptionResNetV1, InceptionResNetV2

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

    def test_inception_resnet_v1_forward_shape(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionResNetV1(INPUT_CHANNEL, NUM_CLASS)

        y = m(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS))

    def test_inception_resnet_v2_forward_shape(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionResNetV2(INPUT_CHANNEL, NUM_CLASS)

        y = m(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS))
