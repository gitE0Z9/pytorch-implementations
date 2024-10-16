from unittest import TestCase

import torch
from torch.testing import assert_close

from ..models.textcnn import TextCNN


class TestModel(TestCase):
    def test_output_shape(self):
        """test output shape"""
        model = TextCNN(26, 8, output_size=10)

        x = torch.randint(0, 26, (1, 5))
        output = model(x)

        assert_close(output.shape, torch.Size((1, 10)))
