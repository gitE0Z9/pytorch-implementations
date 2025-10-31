import torch

from ..models.dcgan import DCGANDiscriminator, DCGANGenerator

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 32
LATENT_DIM = 16
HIDDEN_DIM = 64


class TestModel:
    def test_dcgan_generator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, LATENT_DIM))

        # 2 -> 32 => upscale #block times
        # 1024 -> 64
        model = DCGANGenerator(
            LATENT_DIM,
            INPUT_CHANNEL,
            HIDDEN_DIM * (2**4),
            num_block=4,
            init_shape=(2, 2),
        )

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )

    def test_dcgan_discriminator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        # 32 -> 1 => downscale (1 + #block) times
        # 64 -> 1024
        model = DCGANDiscriminator(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
            num_block=4,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, 1))
