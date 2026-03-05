import torch

from ..models.unet.model import UNet
from ..models.unet.network import ConvBlock, DownSampling, UpSampling

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = 32
NUM_CLASS = 21
HIDDEN_DIM = 8


class TestNetwork:
    def test_conv_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = ConvBlock(INPUT_CHANNEL, HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_downsampling_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = DownSampling(HIDDEN_DIM + 1, HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )

    def test_upsampling_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )
        z = torch.rand((BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = UpSampling(HIDDEN_DIM + 1, HIDDEN_DIM + 2)

        y = model(x, z)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    def test_unet_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = UNet(output_size=NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
