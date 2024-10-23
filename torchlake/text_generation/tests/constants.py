from torchlake.common.schemas.nlp import NlpContext

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 10
EMBED_DIM = 8
HIDDEN_DIM = 8
NUM_CLASS = 5
CONTEXT = NlpContext(device="cpu", max_seq_len=SEQ_LEN)
