from collections import Counter, defaultdict
from functools import reduce
from typing import Literal
from itertools import product

import torch


class CoOccurrenceCounter:

    def __init__(self, vocab_size: int, padding_idx: int | None = None):
        """word context co-occurrence matrix"""
        super(CoOccurrenceCounter, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.counts: Counter[tuple[int, int]] = Counter()

    def update_counts(self, gram: torch.Tensor, context: torch.Tensor):
        """update counts of (word, context)

        Args:
            gram (torch.Tensor): a center word
            context (torch.Tensor): context surround a center word
        """
        if self.padding_idx is not None:
            filter_idx = gram != self.padding_idx
            gram = gram[filter_idx]
            context = context[filter_idx[:, 0]]

        # position encoding
        context += torch.arange(0, context.size(1)).view(1, -1) * self.vocab_size

        for gram, context_tokens in zip(gram.squeeze().tolist(), context.tolist()):
            self.counts.update(product([gram], context_tokens))

    def get_context_counts(self) -> dict[tuple[str], int]:
        """_summary_

        Returns:
            dict[tuple[str], int]: _description_
        """
        output = Counter()

        for (_, c), count in self.counts.items():
            output.update({c: count})

        return output

    def get_pair_counts(
        self,
        key_by: Literal["gram", "context"] | None = None,
    ) -> dict[tuple[str, tuple[str]], int]:
        """_summary_

        Args:
            key_by (Literal["gram", "context"] | None, optional): _description_. Defaults to None.

        Returns:
            dict[tuple[str, tuple[str]], int]: _description_
        """

        if key_by is None:
            return self.counts

        output = defaultdict(dict)

        # early return to prevent iterate over counts
        if key_by not in ["gram", "context"]:
            return output

        for (gram, context), count in self.counts.items():
            if key_by == "gram":
                output[gram][context] = count
            elif key_by == "context":
                output[context][gram] = count

        return output
