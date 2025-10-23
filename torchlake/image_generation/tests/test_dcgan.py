import torch

from ..models.dcgan import DCGANDiscriminator, DCGANGenerator


class TestDCGANGenerator:
    def test_forward_shape(self):
        x = torch.rand((1, 16))

        # 4 -> 32 => upscale 3 times
        # 512 -> 64
        model = DCGANGenerator(16, 3, 64, num_block=3, init_shape=(4, 4))

        y = model(x)

        assert y.shape == torch.Size((1, 3, 32, 32))


class TestDCGANDiscriminator:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        # 32 -> 4 => downscale 3 times
        # 64 -> 512
        model = DCGANDiscriminator(3, 64, image_shape=(32, 32), num_block=3)

        y = model(x)

        assert y.shape == torch.Size((1, 1))
