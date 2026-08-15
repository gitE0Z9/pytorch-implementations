import random
from typing import Literal, Sequence

import nltk
from nltk.corpus import wordnet as wn
from torch import nn

import torchtext


def _ensure_wordnet() -> None:
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        # nltk.download("omw-1.4") # Open Multilingual Wordnet


def _truncated_geometric(n: int, param: float) -> int:
    """Sample k in {0, ..., n-1} with P[k] proportional to param**k."""
    if n <= 1:
        return 0
    weights = [param**i for i in range(n)]
    return random.choices(range(n), weights=weights, k=1)[0]


class SynonymReplacement(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        q: float = 0.5,
        pos_tags: Sequence[Literal["NOUN", "VERB", "ADJ", "ADV"]] = (
            "NOUN",
            "VERB",
            "ADJ",
            "ADV",
        ),
        vocab: torchtext.vocab.Vocab | None = None,
        is_vocab_lowered: bool = False,
    ):
        """Replace word with synonym as an augmentation in [1509.01626]

        Args:
            p (float, optional): probability of how many words to replace. Defaults to 0.5.
            q (float, optional): probability of which synonym to pick. Defaults to 0.5.
            pos_tags (Sequence[Literal["NOUN", "VERB", "ADJ", "ADV"]], optional): pos of synonyms candidates. Defaults to ( "NOUN", "VERB", "ADJ", "ADV", ).
            vocab (torchtext.vocab.Vocab | None, optional): set the vocabulary to filter unknow words. Defaults to None.
            is_vocab_lowered (bool, optional): are words in the vocabulary lowered. Defaults to False.
        """
        self.p = p
        self.q = q
        self.is_vocab_lowered = is_vocab_lowered

        super().__init__()
        _ensure_wordnet()
        _WN_POS = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
        self.pos_tags = [_WN_POS[t] for t in pos_tags]
        self.vocab = vocab

    def _is_in_vocab(self, word: str) -> bool:
        if self.is_vocab_lowered:
            word = word.lower()

        return word in self.vocab

    def _synonyms(self, word: str) -> list[str]:
        seen = {word.lower()}
        output = []
        for pos in self.pos_tags:
            for syn in wn.synsets(word, pos=pos):
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    key = name.lower()
                    if key in seen or (
                        self.vocab is not None and not self._is_in_vocab(name)
                    ):
                        continue
                    seen.add(key)
                    output.append(key if self.is_vocab_lowered else name)
        return output

    def forward(self, x: list[str]) -> list[str]:
        """forward

        Args:
            x (list[str]): tokenized string

        Returns:
            list[str]: replaced string
        """
        x = list(x)

        replaceables = [
            (i, token)
            for i, token in enumerate(x)
            if token.isalpha() and self._synonyms(token)
        ]

        r = _truncated_geometric(len(replaceables) + 1, self.p)
        if r == 0:
            return x
        chosen = random.sample(replaceables, r)

        for i, prev_token in chosen:
            synonyms = self._synonyms(prev_token)
            if not synonyms:
                continue
            s = _truncated_geometric(len(synonyms), self.q)
            x[i] = synonyms[s]

        return x
