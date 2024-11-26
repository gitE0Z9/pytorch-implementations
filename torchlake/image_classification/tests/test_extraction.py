import torch
from torch.testing import assert_close

from ..models.extraction import Extraction, ExtractionFeatureExtractor


class TestExtraction:
    def test_output_shape(self):
        model = Extraction(output_size=1000)
        x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, 1000]))


class TestExtractionFeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    def test_output_shape(self):
        self.setUp()

        model = ExtractionFeatureExtractor("block")

        y: torch.Tensor = model.forward(self.x, ["0_1", "1_1", "2_1", "3_1", "output"])
        for dim, scale in zip(
            [192, 512, 1024, 1024],
            [28, 14, 7, 7],
        ):
            assert y.pop(0).shape == torch.Size((1, dim, scale, scale))

        assert y.pop().shape == torch.Size((1, 1024))
