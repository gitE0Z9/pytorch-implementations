import random
import unittest

import pytest

from ..helpers.vocab import Vocab, CharNgramVocab
from ..schemas.nlp import NLPContext


class TestVocab:
    def test_build_vocab(self):
        sentences = [
            ["there", "are", "five", "birds", "on", "the", "field", "."],
            ["there", "are", "two", "birds", "eating", "worms", "."],
        ]
        vocab = Vocab(context=NLPContext(min_frequency=1))
        vocab.build_from_iterator(iter(sentences))

        expected = [
            "there",
            "are",
            "two",
            "five",
            "birds",
            "on",
            "the",
            "field",
            "eating",
            "worms",
            ".",
        ]
        for token in expected:
            assert token in vocab

    @pytest.mark.parametrize("max_tokens", [None, 100])
    def test_build_vocab_with_pressure(self, max_tokens: int):
        """pressure and smoke test"""
        context = NLPContext(min_frequency=1, max_tokens=max_tokens)

        def gen_sentence():
            for _ in range(1_000_000):
                yield [
                    str(random.randint(0, 10_000)) for _ in range(context.max_seq_len)
                ]

        vocab = Vocab(context)
        vocab.build_from_iterator(gen_sentence())

        if max_tokens is not None:
            assert len(vocab) <= max_tokens + len(context.special_tokens)


class TestCharNgramVocab(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = CharNgramVocab(context=NLPContext(min_frequency=0))

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
