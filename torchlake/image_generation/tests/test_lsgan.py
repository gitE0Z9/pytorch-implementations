import torch

from ..models.lsgan import LSGANDiscriminator, LSGANGenerator

BATCH_SIZE = 1
INPUT_CHANNEL = 3
LATENT_DIM = 16
HIDDEN_DIM = 256


class TestModel:
    def test_lsgan_generator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, LATENT_DIM))

        # 7 -> 112 => upscale (2 + #block) times
        # 256 -> 64
        model = LSGANGenerator(
            LATENT_DIM,
            INPUT_CHANNEL,
            HIDDEN_DIM,
            num_block=2,
            init_shape=(7, 7),
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, INPUT_CHANNEL, 112, 112))

    def test_lsgan_discriminator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, 112, 112))

        # 112 -> 7 => downscale (1 + #block) times
        # 64 -> 512
        model = LSGANDiscriminator(
            INPUT_CHANNEL,
            64,
            image_shape=(112, 112),
            num_block=3,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, 1))
