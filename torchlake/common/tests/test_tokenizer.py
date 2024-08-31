import unittest

from torchtext.data.utils import get_tokenizer

from ..helpers.tokenizer import CharNgramTokenizer
from ..helpers.vocab import CharNgramVocab


class TestNgramTokenizer(unittest.TestCase):

    def setUp(self) -> None:
        self.candidate = "word"
        self.tokenizer = CharNgramTokenizer(
            get_tokenizer("basic_english"),
            CharNgramVocab(),
            [2],
        )

    def test_tokenize(self):
        tokenized, indices = self.tokenizer(self.candidate)

        self.assertEqual(tokenized, ["<wo", "or", "rd>"])
        self.assertEqual(
            indices,
            self.tokenizer.vocab.word_vocab.lookup_indices([self.candidate]) * 3,
        )
