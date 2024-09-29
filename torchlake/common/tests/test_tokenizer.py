import unittest

from torchtext.data.utils import get_tokenizer

from ..helpers.tokenizer import CharNgramTokenizer
from ..helpers.vocab import CharNgramVocab
from ..schemas.nlp import NlpContext


class TestNgramTokenizer(unittest.TestCase):

    def setUp(self) -> None:
        self.candidate = "word"
        self.tokenizer = CharNgramTokenizer(
            get_tokenizer("basic_english"),
            CharNgramVocab(context=NlpContext(min_frequency=0)),
            [2],
        )

    def test_tokenize(self):
        subwords, words, word_spans = self.tokenizer(self.candidate)

        self.assertEqual(subwords, ["<wo", "or", "rd>"])
        self.assertEqual(
            words,
            self.tokenizer.vocab.lookup_indices([self.candidate]),
        )
        self.assertEqual(word_spans, [3])
        self.assertEqual(len(subwords), word_spans[0])
