from typing import Sequence


from ..dcgan.model import DCGANDiscriminator
from torchlake.common.models.model_base import ModelBase


class ACGANDiscriminator(DCGANDiscriminator):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        hidden_dim: int = 64,
        image_shape: Sequence[int] = (32, 32),
        num_block: int = 4,
    ):
        self.hidden_dim = hidden_dim
        self.num_block = num_block

        assert len(image_shape) == 2, "image size must be (height, width)"
        for size in image_shape:
            assert (
                size % (2**num_block) == 0
            ), "image size must be divisible by 2^num_block"

        self.init_image_shape = image_shape
        self.final_image_shape = tuple(
            size // (2 ** (self.num_block + 1)) for size in self.init_image_shape
        )
        ModelBase.__init__(self, input_channel, 1 + output_size)
