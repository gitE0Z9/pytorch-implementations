import torch
from torch.testing import assert_close

from torchlake.common.schemas.nlp import NlpContext

from ..utils.decode import viterbi_decode

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 10
EMBED_DIM = 8
HIDDEN_DIM = 8
NUM_CLASS = 5
CONTEXT = NlpContext(device="cpu", max_seq_len=SEQ_LEN)


class TestViterbiDecode:
    def test_shape(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        transition_matrix = torch.rand(NUM_CLASS, NUM_CLASS)

        path, score = viterbi_decode(x, transition_matrix, context=CONTEXT)

        assert path.shape == torch.Size((BATCH_SIZE, SEQ_LEN))
        assert score.shape == torch.Size((BATCH_SIZE,))

    def test_example(self):
        batch_size = 1
        max_seq_len = 4
        CONTEXT.max_seq_len = max_seq_len
        x = torch.Tensor(
            [
                [
                    [-10, 10, -10, -10, -10, -10],  # t=1
                    [-10, -10, -10, -10, 10, -10],  # t=2
                    [-10, -10, -10, -10, -10, 10],  # t=3
                    [-10, -10, 10, -10, -10, -10],  # t=4
                ]
            ]
        )
        # two states
        # unk, bos, eos, pad, token1, token2
        t = torch.Tensor(
            [
                [-10, -10, -10, -10, 10, -10],
                # tune higher from bos to token1, so transfer to 4
                # if token1 and token2 are evenly possible, test will have undeterministic result
                # and a tiny difference in prior could cause highly divergent result (╯°□°）╯︵ ┻━┻
                [-10, -10, -10, -10, 10, -10],
                [-10, -10, -10, 10, -10, -10],
                [-10, -10, -10, 10, -10, -10],
                [-10, -10, -10, -10, -10, 10],
                [-10, -10, 10, -10, -10, -10],
            ]
        )
        best_path, best_score = viterbi_decode(x, t, context=CONTEXT)
        assert best_path.shape == torch.Size((batch_size, max_seq_len))
        assert best_score.shape == torch.Size((batch_size,))
        assert_close(best_path, torch.LongTensor([[1, 4, 5, 2]]))
