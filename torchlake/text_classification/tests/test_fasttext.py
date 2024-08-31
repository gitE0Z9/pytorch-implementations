from unittest import TestCase

import torch
from torch.testing import assert_close

from ..models import FastText


class TestModel(TestCase):
    def test_words_vector_shape(self):
        """test output shape"""
        model = FastText(10, 8, 10)

        x = torch.randint(0, 10, (1, 5))
        indices = torch.randint(0, 3, (1, 5, 8))
        output = model.get_words_vector(x, indices)

        assert_close(output.shape, torch.Size((1, 3, 8)))

    def test_sentence_vector_shape(self):
        """test output shape"""
        model = FastText(10, 8, 10)

        x = torch.randint(0, 10, (1, 5))
        indices = torch.randint(0, 3, (1, 5, 8))
        output = model.get_sentence_vector(x, indices)

        assert_close(output.shape, torch.Size((1, 8)))

    def test_output_shape(self):
        """test output shape"""
        model = FastText(10, 8, 10)

        x = torch.randint(0, 10, (1, 5))
        indices = torch.randint(0, 3, (1, 5, 8))
        output = model.forward(x, indices)

        assert_close(output.shape, torch.Size((1, 10)))
