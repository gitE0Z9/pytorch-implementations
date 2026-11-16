from torch import nn

from torchlake.common.models.model_base import ModelBase

from .network import HighwayBlock


class HighwayNetwork(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        config: tuple[tuple[str | int]] | None = None,
    ):
        """Highway network [1505.00387v2]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            config (tuple[tuple[str | int]], optional): layer config, format is (layer_type, input_channel, output_channel, kernel_size), layer_type "c" is highway layer and "p" is max pooling. Defaults to None.
        """
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return self.config[-1][2]

    @property
    def config(self) -> tuple[tuple[str | int]]:
        # layer_type, input_channel, output_channel, kernel_size
        return self._config or (
            ("c", 32, 32, 3),
            ("c", 32, 32, 3),
            ("c", 32, 64, 3),
            ("p", 0, 0, 0),
            ("c", 64, 64, 3),
            ("c", 64, 64, 3),
            ("c", 64, 128, 3),
            ("p", 0, 0, 0),
            ("c", 128, 128, 3),
            ("c", 128, 128, 3),
            ("c", 128, 256, 3),
            ("p", 0, 0, 0),
            ("c", 256, 256, 3),
            ("c", 256, 256, 3),
            ("c", 256, 512, 3),
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            HighwayBlock(input_channel, self.config[0][1], 3),
            nn.MaxPool2d(2, 2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for layer_type, in_c, out_c, kernel_size in self.config:
            if layer_type == "c":
                self.blocks.append(HighwayBlock(in_c, out_c, kernel_size))
            elif layer_type == "p":
                self.blocks.append(nn.MaxPool2d(2, 2))
