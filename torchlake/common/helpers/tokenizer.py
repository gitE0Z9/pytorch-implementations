# TODO: really slooooooooooow~~~
from functools import lru_cache
from typing import Callable


class CharNgramTokenizer:
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
        ngrams: list[int],
        cache_size: int | None = 100_000,
    ) -> None:
        """Characeter n-grams tokenizer

        Args:
            tokenizer (Callable): tokenizer e.g. spacy or any callable to split a sentence into tokens.
            ngrams (list[int]): n-grams size, for instance, [2] means only collect bigrams, [3,4,5] means collect all of trigrams, quatragram, pentagram.
        """
        self.tokenizer = tokenizer
        self.ngrams = tuple(sorted(set(ngrams)))
        self._cache = lru_cache(maxsize=cache_size)(self._compute)

    def _compute(self, token: str) -> tuple[str, ...]:
        wrapped = f"<{token}>"
        n = len(wrapped)
        grams: list[str] = []
        for ngram in self.ngrams:
            if ngram >= n:
                grams.append(wrapped)
                break
            grams.extend([wrapped[i : i + ngram] for i in range(n - ngram + 1)])
        return tuple(grams)

    def __call__(self, sentence: str) -> tuple[list[str], list[str], list[int]]:
        """tokenize sentence into ngrams, words, word_spans

        Args:
            sentence (str): sentence

        Returns:
            tuple[list[str], list[str], list[int]]: ngrams, words, word_spans
        """
        words: list[str] = self.tokenizer(sentence)
        subwords: list[str] = []
        word_spans: list[int] = []
        for token in words:
            grams = self._cache(token)
            subwords.extend(grams)
            word_spans.append(len(grams))

        return subwords, words, word_spans
