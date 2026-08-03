from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional

import torch  # noqa: F401  (must be imported first so _torchtext can find libtorch/libc10)
import torch.nn as nn

from . import _torchtext


class Vocab(nn.Module):
    """Thin nn.Module wrapper around the compiled _torchtext.Vocab, mirroring the
    original torchtext.vocab.Vocab API: v(tokens) / len(v) / v[token] / "tok" in v."""

    def __init__(self, cpp_vocab: "_torchtext.Vocab") -> None:
        super().__init__()
        self.vocab = cpp_vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        return self.vocab.__contains__(token)

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def set_default_index(self, index: Optional[int]) -> None:
        self.vocab.set_default_index(index)

    def get_default_index(self) -> Optional[int]:
        return self.vocab.get_default_index()

    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)

    def append_token(self, token: str) -> None:
        self.vocab.append_token(token)

    def lookup_token(self, index: int) -> str:
        return self.vocab.lookup_token(index)

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return self.vocab.lookup_tokens(indices)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)

    def get_stoi(self) -> Dict[str, int]:
        return self.vocab.get_stoi()

    def get_itos(self) -> List[str]:
        return self.vocab.get_itos()


def vocab(
    ordered_dict: Dict, min_freq: int = 1, specials: Optional[List[str]] = None, special_first: bool = True
) -> Vocab:
    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if special_first:
        tokens[0:0] = specials
    else:
        tokens.extend(specials)

    return Vocab(_torchtext.Vocab(tokens, None))


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> Vocab:
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is None:
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
    else:
        assert len(specials) < max_tokens, "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
        ordered_dict = OrderedDict(sorted_by_freq_tuples[: max_tokens - len(specials)])

    return vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
