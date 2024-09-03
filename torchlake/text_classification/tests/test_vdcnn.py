import pytest
import torch
from torchlake.common.schemas.nlp import NlpContext

from ..models import Vdcnn
from ..models.vdcnn.network import Block


def test_block_output_shape():
    """test output shape"""
    x = torch.rand((4, 16, 1024))

    model = Block(16, 32, 3)
    output = model(x)

    assert output.shape == torch.Size((4, 32, 1024))


@pytest.mark.parametrize("depth_mutliplier", [1, 2, 3, 4])
@pytest.mark.parametrize("enable_shortcut", [True, False])
def test_vdcnn_output_shape(depth_mutliplier: int, enable_shortcut: bool):
    """test output shape"""
    max_seq_len = 1024
    model = Vdcnn(
        70,
        10,
        depth_multipier=depth_mutliplier,
        enable_shortcut=enable_shortcut,
        context=NlpContext(max_seq_len=max_seq_len),
    )

    x = torch.randint(0, 70, (1, max_seq_len))
    output = model(x)

    assert output.shape == torch.Size((1, 10))
