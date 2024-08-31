import unittest
from ..helpers.vocab import CharNgramVocab


class TestCharNgramVocab(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = CharNgramVocab()

    def test_add_word_vocab(self):
        expected = "word"
        self.vocab.add_token(expected)

        self.assertIn(expected, self.vocab.word_vocab)

    def test_add_subword_vocab(self):
        expected = "<wor"
        self.vocab.add_subtoken(expected)

        self.assertIn(expected, self.vocab.subword_vocab)

    def test_hash_subword(self):
        expected = "<wor"
        result = self.vocab.hash_subtoken(expected.encode())

        self.assertIsInstance(result, int)

    def test_lookup_indices(self):
        expected = ["<wor", "wor", "ord>"]
        for subtoken in expected:
            self.vocab.add_subtoken(subtoken)
        result = self.vocab.lookup_indices(expected)

        self.assertEqual(len(result), 3)
        for index in result:
            self.assertIsInstance(index, int)
