import unittest

from torchtext.data.utils import get_tokenizer

from ..helpers.tokenizer import CharNgramTokenizer


class TestNgramTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = "a yellow bird flew over the sky."
        self.tokenizer = CharNgramTokenizer(get_tokenizer("basic_english"), [3])

    def test_tokenize(self):
        subwords, words, word_spans = self.tokenizer(self.candidate)

        self.assertEqual(
            subwords,
            [
                "<a>",
                "<ye",
                "yel",
                "ell",
                "llo",
                "low",
                "ow>",
                "<bi",
                "bir",
                "ird",
                "rd>",
                "<fl",
                "fle",
                "lew",
                "ew>",
                "<ov",
                "ove",
                "ver",
                "er>",
                "<th",
                "the",
                "he>",
                "<sk",
                "sky",
                "ky>",
                "<.>",
            ],
        )
        self.assertEqual(
            words,
            ["a", "yellow", "bird", "flew", "over", "the", "sky", "."],
        )
        self.assertEqual(word_spans, [1, 6, 4, 4, 4, 3, 3, 1])
        self.assertEqual(len(subwords), sum(word_spans))
