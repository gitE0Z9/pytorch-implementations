import torch

from ..models.dcgan import DCGANDiscriminator, DCGANGenerator

BATCH_SIZE = 1
INPUT_CHANNEL = 3
LATENT_DIM = 16


class TestDCGANGenerator:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, LATENT_DIM))

        # 4 -> 32 => upscale 3 times
        # 512 -> 64
        model = DCGANGenerator(
            LATENT_DIM,
            INPUT_CHANNEL,
            LATENT_DIM * (2**4),
            num_block=4,
            init_shape=(2, 2),
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, INPUT_CHANNEL, 32, 32))


class TestDCGANDiscriminator:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        # 32 -> 4 => downscale 3 times
        # 64 -> 512
        model = DCGANDiscriminator(3, 64, image_shape=(32, 32), num_block=3)

        y = model(x)

        assert y.shape == torch.Size((1, 1))
