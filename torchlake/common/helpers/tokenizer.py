# TODO: really slooooooooooow~~~
from typing import Callable


class CharNgramTokenizer:
    def __init__(
        self,
        tokenizer: Callable,
        ngrams: list[int],
    ) -> None:
        """Characeter n-grams tokenizer

        Args:
            tokenizer (Callable): tokenizer e.g. spacy or any callable to split a sentence into tokens.
            ngrams (list[int]): n-grams size, for instance, [2] means only collect bigrams, [3,4,5] means collect all of trigrams, quatragram, pentagram.
        """
        self.tokenizer = tokenizer
        self.ngrams = ngrams

    def __call__(self, sentence: str) -> tuple[list[str], list[int], list[int]]:
        """tokenize sentence into ngrams, words, word_spans

        Args:
            sentence (str): sentence

        Returns:
            tuple[list[str], list[int], list[int]]: ngrams, words, word_spans
        """
        words: list[str] = self.tokenizer(sentence)

        word_spans = []
        subwords = []
        # loop over ngrams combination
        for token in words:
            word_length = len(token)
            word_span = 0
            for ngram in self.ngrams:
                slide_times = max(word_length - ngram + 1, 1)

                # no sliding at all
                if slide_times == 1:
                    sub_token = f"<{token}>"
                    subwords.append(sub_token)
                    word_span += 1
                # sliding is needed
                else:
                    subtokens = [token[i : i + ngram] for i in range(slide_times)]
                    subtokens[0] = "<" + subtokens[0]
                    subtokens[-1] += ">"

                    subwords.extend(subtokens)
                    word_span += len(subtokens)

            word_spans.append(word_span)

        return subwords, words, word_spans
