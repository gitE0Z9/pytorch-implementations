from unittest import TestCase

import torch
from torch.testing import assert_close
from torchlake.common.schemas.nlp import NlpContext

from ..models import CharCnn


class TestModel(TestCase):
    def test_output_shape(self):
        """test output shape"""
        max_seq_len = 27 * 10 + 96
        model = CharCnn(70, 10, NlpContext(device="cpu", max_seq_len=max_seq_len))

        x = torch.randint(0, 70, (1, max_seq_len))
        output = model(x)

        assert_close(output.shape, torch.Size((1, 10)))
