# TODO: really slooooooooooow~~~
from typing import Callable

from .vocab import CharNgramVocab


class CharNgramTokenizer:
    def __init__(
        self,
        tokenizer: Callable,
        vocab: CharNgramVocab,
        ngrams: list[int],
    ) -> None:
        """Characeter n-grams tokenizer

        Args:
            tokenizer (Callable): tokenizer e.g. spacy or any callable to split a sentence into tokens.
            vocab (CharNgramVocab): character ngram vocab, which stored word vocab and cached subword vocab.
            ngrams (list[int]): n-grams size, for instance, [2] means only collect bigrams, [3,4,5] means collect all of trigrams, quatragram, pentagram.
        """
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.ngrams = ngrams

    def __call__(self, sentence: str) -> tuple[list[str], list[int], list[int]]:
        """tokenize sentence into ngrams, words, word_spans

        Args:
            sentence (str): sentence

        Returns:
            tuple[list[str], list[int], list[int]]: ngrams, words, word_spans
        """
        tokens: list[str] = self.tokenizer(sentence)

        # add words to hash cache
        self.vocab.add_subtokens(tokens)

        words = self.vocab.lookup_indices(tokens)
        word_spans = []
        subwords = []
        # loop over ngrams combination
        for token in tokens:
            word_length = len(token)
            word_span = 0
            for ngram in self.ngrams:
                slide_times = max(word_length - ngram + 1, 1)

                # no sliding at all
                if slide_times == 1:
                    sub_token = f"<{token}>"
                    # add a new subword to subword vocab
                    self.vocab.add_subtoken(sub_token)
                    subwords.append(sub_token)
                    word_span += 1
                # sliding is needed
                else:
                    subtokens = [token[i : i + ngram] for i in range(slide_times)]
                    subtokens[0] = "<" + subtokens[0]
                    subtokens[-1] += ">"

                    # add a new subword to subword vocab
                    self.vocab.add_subtokens(subtokens)
                    subwords.extend(subtokens)
                    word_span += len(subtokens)

            word_spans.append(word_span)

        return subwords, words, word_spans
