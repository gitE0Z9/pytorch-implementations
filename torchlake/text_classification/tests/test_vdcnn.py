from unittest import TestCase

import pytest
import torch
from torchlake.common.schemas.nlp import NlpContext

from ..models import Vdcnn
from ..models.vdcnn.network import Block


class TestBlock(TestCase):
    def test_output_shape(self):
        """test output shape"""
        x = torch.rand((4, 16, 1024))

        model = Block(16, 32, 3)
        output = model(x)

        self.assertEqual(output.shape, torch.Size((4, 32, 1024)))


class TestModel(TestCase):
    @pytest.mark.parametrize("depth_mutliplier", [1, 2, 3, 4])
    def test_output_shape(self):
        """test output shape"""
        max_seq_len = 1024
        model = Vdcnn(
            70,
            10,
            context=NlpContext(max_seq_len=max_seq_len),
        )

        x = torch.randint(0, 70, (1, max_seq_len))
        output = model(x)

        self.assertEqual(output.shape, torch.Size((1, 10)))
