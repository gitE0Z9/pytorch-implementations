import torch

from ..models.texture.model import TextureNet

BATCH_SIZE = 1
INPUT_CHANNEL = 3
IMAGE_SIZE = 256
NUM_SCALE_FACTOR = 5


class TestModel:
    def test_texture_net_forward_shape(self):
        stacks = [
            torch.rand(
                (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE // (2**i), IMAGE_SIZE // (2**i))
            )
            for i in range(NUM_SCALE_FACTOR)
        ]
        model = TextureNet(INPUT_CHANNEL, num_scale_factor=NUM_SCALE_FACTOR)

        y = model(stacks)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )
