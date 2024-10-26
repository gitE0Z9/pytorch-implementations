import torch
from torch.testing import assert_close

from ..utils.decode import viterbi_decode
from .constants import BATCH_SIZE, CONTEXT, NUM_CLASS, SEQ_LEN


class TestViterbiDecode:
    def test_shape(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        transition_matrix = torch.rand(NUM_CLASS, NUM_CLASS)

        path, score = viterbi_decode(x, transition_matrix, context=CONTEXT)

        assert path.shape == torch.Size((BATCH_SIZE, SEQ_LEN))
        assert score.shape == torch.Size((BATCH_SIZE,))

    def test_example(self):
        CONTEXT.max_seq_len = 4
        x = torch.Tensor(
            [
                [
                    [-1e2, 1e2, -1e2, -1e2, -1e2, -1e2],  # t=1
                    [-1e2, -1e2, -1e2, -1e2, 1e2, 1e1],  # t=2
                    [-1e2, -1e2, -1e2, -1e2, 1e1, 1e2],  # t=3
                    [-1e2, -1e2, 1e2, -1e2, -1e2, -1e2],  # t=4
                ]
            ]
        )
        # two states
        # unk, bos, eos, pad, token1, token2
        t = torch.Tensor(
            [
                [-1e2, -1e4, -1e2, -1e4, -5e1, -1e2],
                # tune higher from bos to token1, so transfer to 4
                # if token1 and token2 are evenly possible, test will have undeterministic result
                # and a tiny difference in prior could cause highly divergent result (╯°□°）╯︵ ┻━┻
                [-1e2, -1e4, -1e4, -1e4, -5e1, -1e2],
                [-1e4, -1e4, -1e4, 0, -1e4, -1e4],
                [-1e4, -1e4, -1e4, 0, -1e4, -1e4],
                [-1e2, -1e4, -1e2, -1e4, -1e2, -5e1],
                [-1e2, -1e4, -1e2, -1e4, -5e1, -1e2],
            ]
        )
        best_path, best_score = viterbi_decode(x, t, context=CONTEXT)
        assert best_path.shape == torch.Size((1, 4))
        print(best_path)
        assert_close(best_path, torch.LongTensor([[1, 4, 5, 2]]))
        assert best_score.shape == torch.Size((1,))
