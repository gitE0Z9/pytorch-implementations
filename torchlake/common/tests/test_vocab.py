import unittest

from ..helpers.vocab import CharNgramVocab
from ..schemas.nlp import NlpContext


class TestCharNgramVocab(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = CharNgramVocab(context=NlpContext(min_frequency=0))

    def test_build_word_vocab(self):
        expected = [["word", "is", "large"]]
        self.vocab.build_word_vocab(iter(expected))

        for ele in expected[0]:
            self.assertIn(ele, self.vocab.word_vocab)

    def test_add_subword_vocab(self):
        expected = "<wor"
        self.vocab.add_subtoken(expected)

        self.assertIn(expected, self.vocab.subword_vocab)

    def test_add_subtokens(self):
        expected = ["<wor", "orl", "rld>"]
        self.vocab.add_subtokens(expected)

        for ele in expected:
            self.assertIn(ele, self.vocab.subword_vocab)

    def test_hash_subword(self):
        expected = "<wor"
        result = self.vocab.hash_subtoken(expected)

        self.assertIsInstance(result, int)

    def test_lookup_indices(self):
        expected = ["<wor", "wor", "ord>"]
        for subtoken in expected:
            self.vocab.add_subtoken(subtoken)
        result = self.vocab.lookup_indices(expected)

        self.assertEqual(len(result), 3)
        for index in result:
            self.assertIsInstance(index, int)
