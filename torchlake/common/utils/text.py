import random
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
    device: torch.device = "cpu",
) -> torch.LongTensor:
    """unigram counts, i.e. word counts, for iterable tensors

    Args:
        data (Iterable[torch.Tensor]): tokens in tensor.
        vocab_size (int): size of vocabulary.
        freq (bool, optional): return in frequency. Defaults to False.
        device (torch.device, optional): device for tensor. Defaults to "cpu".

    Returns:
        torch.LongTensor: unigram counts
    """
    word_counts = torch.zeros(vocab_size, dtype=torch.long).to(device)
    for item in data:
        item = item.to(device)
        word_counts += torch.bincount(item, minlength=vocab_size).to(device)

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


def get_context(
    batch: list[torch.Tensor],
    left_context_size: int = 2,
    right_context_size: int = 2,
    enable_random_context_size: bool = False,
    enable_symmetric_context: bool = True,
    flatten_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get context words for a center word

    Args:
        batch: shape(batch_size, max_seq_len)
        left_context_size (int, optional): left side of a center word. Defaults to 2.
        right_context_size (int, optional): right side of a center word. Defaults to 2.
        enable_random_context_size (bool, optional): enable random context size from 1 to maximum. Defaults to False.
        enable_symmetric_context (bool, optional): disable this variable to disable symmetry check. Defaults to True.
        flatten_output (bool, optional): flatten output to gram(batch_size*seq_len, 1) and context(batch_size*seq_len, neighbor_size). Defaults to False.

    Returns:
        gram,context: shape(batch_size, 1 or context_size - 1, max_seq_len - context_size + 1)
    """
    maximum_context_size = left_context_size + right_context_size + 1
    assert maximum_context_size > 1, "context size should be larger than 0"

    # Check symmetry first
    if enable_symmetric_context:
        assert (
            left_context_size == right_context_size
        ), "asymmetric context is only supported when enable_symmetric_context is False."

        # Context is an odd that falls between 3 ~ maximum_context_size
        if maximum_context_size < 3:
            raise ValueError("contextss size is too small.")
        elif maximum_context_size % 2 == 0:
            raise ValueError("context size should be odd.")

    # for now, only symmetric context supports random mode
    context_size = (
        random.choice(range(3, maximum_context_size + 1, 2))
        if enable_random_context_size and enable_symmetric_context
        else maximum_context_size
    )

    half_size = context_size // 2 if enable_symmetric_context else left_context_size
    context_indice = list(range(context_size))
    context_indice.pop(half_size)

    batch: torch.Tensor = torch.stack(batch)
    batch = batch.unfold(1, context_size, 1)
    gram = batch[:, :, half_size].long().unsqueeze(1)
    context = batch[:, :, context_indice].long()

    if flatten_output:
        return (
            # batch_size*seq_len, 1
            gram.transpose(-1, -2).flatten(0, 1),
            # batch_size*seq_len, neighbor_size
            context.flatten(0, 1),
        )

    return (
        # batch_size, 1, seq_len
        gram,
        # batch_size, neighbor_size, seq_len
        context.transpose(-1, -2),
    )
