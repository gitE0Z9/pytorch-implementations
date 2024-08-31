from typing import Callable

from .vocab import CharNgramVocab


class CharNgramTokenizer:
    def __init__(
        self,
        tokenizer: Callable,
        vocab: CharNgramVocab,
        ngrams: list[int],
        enable_mutable_vocab: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.ngrams = ngrams
        self.enable_mutable_vocab = enable_mutable_vocab

    def activate_vocab(self):
        self.enable_mutable_vocab = True

    def frozen_vocab(self):
        self.enable_mutable_vocab = False

    def __call__(self, sentence: str) -> tuple[list[str], list[int]]:
        subwords, word_indices = [], []

        tokenized: list[str] = self.tokenizer(sentence)
        for token in tokenized:
            if self.enable_mutable_vocab:
                self.vocab.add_token(token)

            for ngram in self.ngrams:

                cloned_token = token
                slide_times = len(cloned_token) - ngram + 1

                if slide_times == 1:
                    sub_token = "<" + cloned_token + ">"

                    if self.enable_mutable_vocab:
                        self.vocab.add_subtoken(sub_token)

                    subwords.append(sub_token)
                    word_indices.append(self.vocab.get_word_index(token))
                else:
                    for i in range(slide_times):
                        sub_token = cloned_token[i : i + ngram]
                        if i == 0:
                            sub_token = "<" + sub_token
                        if i == slide_times - 1:
                            sub_token += ">"

                        if self.enable_mutable_vocab:
                            self.vocab.add_subtoken(sub_token)

                        subwords.append(sub_token)
                        word_indices.append(self.vocab.get_word_index(token))

        return subwords, word_indices
