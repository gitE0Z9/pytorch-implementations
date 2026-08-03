import heapq
from collections import Counter
from functools import lru_cache
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Iterable

from torchtext.vocab import vocab

from ..schemas.nlp import NLPContext
from ..utils.file import read_json_file, write_json_file
from ..utils.hash import fnv1a_hash
from ..utils.text import build_vocab


class Vocab:
    def __init__(
        self,
        context: NLPContext,
    ):
        self._vocab = {}
        self.context = context

        self.add_token(context.unk_str, context.unk_idx)
        self.add_token(context.bos_str, context.bos_idx)
        self.add_token(context.eos_str, context.eos_idx)
        self.add_token(context.pad_str, context.padding_idx)

    def __len__(self) -> int:
        return len(self._vocab)

    def __getitem__(self, token: str) -> int:
        return self._vocab[token]

    def __contains__(self, v: str) -> bool:
        return v in self._vocab

    def get_stoi(self) -> dict[str, int]:
        return self._vocab

    def get_itos(self) -> dict[int, str]:
        return {i: s for s, i in self._vocab.items()}

    def add_token(self, token: str, index: int | None = None):
        if token in self._vocab:
            return

        if index is not None:
            self._vocab[token] = index
        else:
            self._vocab[token] = len(self._vocab)

    def add_tokens(self, tokens: list[str]):
        for token in tokens:
            self.add_token(token)

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        """retrieve index of a list of tokens from vocab

        Args:
            tokens (list[str]):  a list of tokens

        Returns:
            list[int]: a list of tokens in vocab
        """
        return [self._vocab.get(token, self.context.unk_idx) for token in tokens]

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        """retrieve token by index

        Args:
            indices (list[int]):  a list of indices

        Returns:
            list[str]: a list of tokens
        """
        indices = itemgetter(*indices)(self._vocab)
        return [indices] if isinstance(indices, int) else list(indices)

    def build_from_iterator(self, data: Iterable[list[str]]):
        counter = Counter(chain.from_iterable(data))

        # O(v)
        tokens = [it for it in counter.items() if it[1] >= self.context.min_frequency]

        if self.context.max_tokens:
            # O(vlogk)
            tokens = heapq.nlargest(
                self.context.max_tokens,
                tokens,
                key=lambda it: it[1],
            )

        self.add_tokens(map(lambda it: it[0], tokens))


class CharNgramVocab:
    def __init__(
        self,
        context: NLPContext,
        bucket_size: int = 2 * 10**6,
        encoding: str = "utf-8",
    ) -> None:
        """character ngram vocab

        Args:
            context NLPContext: NLP context.
            bucket_size (int, optional): size of hash bucket. Defaults to 2*10**6.
            encoding (str, optional): encoding of subword string. Defaults to "utf-8".
        """
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.context = context

        # as vocab dict
        self.word_vocab = None
        # as a cache
        self.subword_vocab = {}

    def __len__(self) -> int:
        return len(self.subword_vocab)

    def __getitem__(self, subword: str) -> int:
        return self.subword_vocab[subword]

    def add_subtoken(self, subtoken: str):
        """add subtoken to subword vocab

        Args:
            subtoken (str): ngram token
        """
        if subtoken not in self.subword_vocab:
            self.subword_vocab[subtoken] = self.hash_subtoken(subtoken)

    def add_subtokens(self, subtokens: list[str]):
        """add subtokens to subword vocab

        Args:
            subtokens (list[str]): ngram tokens
        """
        new_subtokens = {
            subtoken: self.hash_subtoken(subtoken)
            for subtoken in set(subtokens)
            if subtoken not in self.subword_vocab
        }
        self.subword_vocab.update(new_subtokens)

    @lru_cache(maxsize=8192)
    def hash_subtoken(self, subtoken: str) -> int:
        """hash a ngram token to index in subword vocab by fnv-1a

        Args:
            subtoken (str): ngram token

        Returns:
            int: hashed index of subtoken
        """
        subtoken = subtoken.encode(self.encoding)
        return fnv1a_hash(subtoken) % self.bucket_size

    def build_word_vocab(self, data: Iterable):
        """add tokens to word vocab

        Args:
            data (Iterable): data
        """
        self.word_vocab = build_vocab(data, self.context)

    def lookup_word_indices(self, words: list[str]) -> list[int]:
        """retrieve index of a list of words from word vocab

        Args:
            words (list[str]): a list of words

        Returns:
            int: indices of a list of words in word vocab
        """
        return self.word_vocab.lookup_indices(words)

    def lookup_indices(self, subtokens: list[str]) -> list[int]:
        """retrieve index of a list of subtokens from subword vocab

        Args:
            subtokens (list[str]):  a list of ngrams tokens

        Returns:
            list[int]: a list of subtokens in subword vocab
        """
        indices = itemgetter(*subtokens)(self.subword_vocab)
        return [indices] if isinstance(indices, int) else list(indices)

    def save_word_vocab(self, path: Path | str):
        """save word vocab to json file

        Args:
            path (Path | str): path to json file
        """
        write_json_file(path, self.word_vocab.get_stoi())

    def save_subword_vocab(self, path: Path | str):
        """save subword vocab to json file

        Args:
            path (Path | str): path to json file
        """
        write_json_file(path, self.subword_vocab)

    def load_word_vocab(self, path: Path | str):
        """load json file of word vocab

        Args:
            path (Path | str): path to json file
        """
        data = read_json_file(path)
        self.word_vocab = vocab(
            data,
            min_freq=0,
            specials=self.context.special_tokens,
        )

    def load_subword_vocab(self, path: Path | str):
        """load json file of subword vocab

        Args:
            path (Path | str): path to json file
        """
        data = read_json_file(path)
        self.subword_vocab = data
