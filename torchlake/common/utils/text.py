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


def get_unigram_counts(
    data: Iterable,
    vocab_size: int,
    freq: bool = False,
) -> torch.LongTensor:
    """unigram counts, i.e. word counts, for iterable tokens

    Args:
        data (Iterable): tokens.
        vocab_size (int): size of vocabulary.
        freq (bool, optional): return in frequency. Defaults to False.

    Returns:
        torch.LongTensor: unigram counts
    """
    counter = Counter()
    for item in data:
        counter.update(item)

    word_counts = [counter.get(i, 0) for i in range(vocab_size)]
    word_counts = torch.LongTensor(word_counts)

    assert len(word_counts) == vocab_size, "Word counts misaligned with vocab."

    if freq:
        return word_counts / sum(word_counts)

    return word_counts


def get_unigram_counts_by_tensor(
    data: Iterable[torch.Tensor],
    vocab_size: int,
    freq: bool = False,
) -> torch.LongTensor:
    """unigram counts, i.e. word counts, for iterable tensors

    Args:
        data (Iterable[torch.Tensor]): tokens in tensor.
        vocab_size (int): size of vocabulary.
        freq (bool, optional): return in frequency. Defaults to False.

    Returns:
        torch.LongTensor: unigram counts
    """
    word_counts = torch.zeros(vocab_size, dtype=torch.long)
    for item in data:
        word_counts += torch.bincount(item, minlength=vocab_size)

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
