from collections import Counter
from typing import Iterable

import torch
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


def get_unigram_counts(data: Iterable, vocab_size: int, freq: bool = False):
    counts = Counter()
    for item in data:
        counts.update(item)

    word_counts = [counts.get(i, 0) for i in range(vocab_size)]
    word_counts = torch.LongTensor(word_counts)

    assert len(word_counts) == vocab_size, "Word counts misaligned with vocab."

    if freq:
        return word_counts / sum(word_counts)

    return word_counts


def is_corpus_title(text: str) -> bool:
    return text.startswith("=") and text.endswith("=")


def is_longer_text(text: str, length: int = 0) -> bool:
    return len(text) > length


def drop_keywords(tokens: list[str], keywords: list[str]) -> list[str]:
    return [token for token in tokens if token not in keywords]
