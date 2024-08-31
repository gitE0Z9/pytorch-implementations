from operator import itemgetter
from pathlib import Path

from torchtext.vocab import vocab

from ..schemas.nlp import NlpContext
from ..utils.file import read_json_file, write_json_file
from ..utils.hash import fnv1a_hash


class CharNgramVocab:
    def __init__(
        self,
        context: NlpContext = NlpContext(),
        encoding: str = "utf-8",
        bucket_size: int = 2 * 10**6,
    ) -> None:
        self.context = context
        self.encoding = encoding
        self.bucket_size = bucket_size

        self.word_vocab = vocab({})
        self.subword_vocab = {}

    def __len__(self) -> int:
        return len(self.subword_vocab)

    def __getitem__(self, subword: str) -> int:
        return self.subword_vocab[subword]

    def add_token(self, token: str):
        if token not in self.word_vocab:
            self.word_vocab.append_token(token)

    def add_subtoken(self, subtoken: str):
        if subtoken not in self.subword_vocab:
            self.subword_vocab[subtoken] = self.hash_subtoken(
                subtoken.encode(self.encoding)
            )

    def hash_subtoken(self, subtoken: bytes) -> str:
        return fnv1a_hash(subtoken) % self.bucket_size

    def get_word_index(self, word: str) -> int:
        return self.word_vocab[word]

    def lookup_indices(self, subtokens: list[str]) -> list[int]:
        return list(itemgetter(*subtokens)(self.subword_vocab))

    def save_word_vocab(self, path: Path | str):
        write_json_file(path, self.word_vocab.get_stoi())

    def save_subword_vocab(self, path: Path | str):
        write_json_file(path, self.subword_vocab)

    def load_word_vocab(self, path: Path | str):
        data = read_json_file(path)
        self.word_vocab = vocab(data)

    def load_subword_vocab(self, path: Path | str):
        data = read_json_file(path)
        self.subword_vocab = data
