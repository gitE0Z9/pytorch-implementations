import torch

from ..utils.decode import viterbi_decode
from .constants import BATCH_SIZE, NUM_CLASS, SEQ_LEN


class TestViterbiDecode:
    def test_shape(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        transition_matrix = torch.rand(NUM_CLASS, NUM_CLASS)

        path, score = viterbi_decode(x, transition_matrix)

        assert path.shape == torch.Size((BATCH_SIZE, SEQ_LEN))
        assert score.shape == torch.Size((BATCH_SIZE,))
