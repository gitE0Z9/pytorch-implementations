from typing import Iterable

from torchtext.vocab import Vocab, build_vocab_from_iterator

from ..schemas import NlpContext


def build_vocab(data: Iterable, context: NlpContext = NlpContext()) -> Vocab:
    vocab = build_vocab_from_iterator(
        data,
        specials=context.special_tokens,
        min_freq=context.min_frequency,
        max_tokens=context.max_tokens,
    )

    vocab.set_default_index(context.unk_idx)

    return vocab
