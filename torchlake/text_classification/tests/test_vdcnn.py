import pytest
import torch
from torchlake.common.schemas.nlp import NLPContext

from ..models.vdcnn import VDCNN
from ..models.vdcnn.network import Block

BATCH_SIZE = 4
VOCAB_SIZE = 26
OUTPUT_SIZE = 10
MAX_SEQ_LEN = 256


class TestNetwork:
    def test_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 16, MAX_SEQ_LEN))

        model = Block(16, 32, 3)
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, 32, MAX_SEQ_LEN))


class TestModel:
    @pytest.mark.parametrize("depth_mutliplier", [1, 2, 3, 4])
    @pytest.mark.parametrize("enable_shortcut", [True, False])
    def test_vdcnn_forward_shape(self, depth_mutliplier: int, enable_shortcut: bool):
        max_seq_len = 1024
        model = VDCNN(
            VOCAB_SIZE,
            OUTPUT_SIZE,
            depth_multipier=depth_mutliplier,
            enable_shortcut=enable_shortcut,
            context=NLPContext(max_seq_len=max_seq_len),
        )

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, max_seq_len))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
