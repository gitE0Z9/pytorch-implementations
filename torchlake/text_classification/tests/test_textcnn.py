from unittest import TestCase

import torch
from torch.testing import assert_close

from ..models import TextCnn


class TestModel(TestCase):
    def test_output_shape(self):
        """test output shape"""
        model = TextCnn(26, 8, 10)

        x = torch.randint(0, 26, (1, 5))
        output = model(x)

        assert_close(output.shape, torch.Size((1, 10)))
