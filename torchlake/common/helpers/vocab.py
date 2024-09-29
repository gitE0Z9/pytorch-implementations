from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Iterable

from torchtext.vocab import vocab

from ..schemas.nlp import NlpContext
from ..utils.file import read_json_file, write_json_file
from ..utils.hash import fnv1a_hash
from ..utils.text import build_vocab


class CharNgramVocab:
    def __init__(
        self,
        bucket_size: int = 2 * 10**6,
        encoding: str = "utf-8",
        context: NlpContext = NlpContext(),
    ) -> None:
        """character ngram vocab

        Args:
            bucket_size (int, optional): size of hash bucket. Defaults to 2*10**6.
            encoding (str, optional): encoding of subword string. Defaults to "utf-8".
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
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

    def build_word_vocab(self, data: Iterable):
        """add tokens to word vocab

        Args:
            data (Iterable): data
        """
        self.word_vocab = build_vocab(data, self.context)

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
