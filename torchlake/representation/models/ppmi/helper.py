from collections import Counter, defaultdict
from typing import Literal

import torch


class CoOccurrenceCounter:

    def __init__(self, vocab_size: int, padding_idx: int | None = None):
        """word-context co-occurrence counter

        Args:
            vocab_size (int): size of vocabulary
            padding_idx (int | None, optional): index of padding token. Defaults to None.
        """
        super(CoOccurrenceCounter, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.counts: Counter[tuple[int, int]] = Counter()

    def update_counts(self, gram: torch.Tensor, context: torch.Tensor):
        """update counts of (word, context)

        Args:
            gram (torch.Tensor): a center word, in shape of (batch*subseq_len, 1)
            context (torch.Tensor): context surround a center word, in shape of (batch*subseq_len, neighbor_size)
        """
        if self.padding_idx is not None:
            filter_idx = (gram != self.padding_idx)[:, 0]
            gram = gram[filter_idx]
            context = context[filter_idx]

        # position encoding
        context += torch.arange(0, context.size(1)).view(1, -1) * self.vocab_size
        gram = gram.repeat_interleave(context.size(1), 1).view(-1).tolist()
        context = context.view(-1).tolist()

        self.counts.update(zip(gram, context))

    def get_context_counts(self) -> dict[int, int]:
        """get context-as-key counts dict, count represent how many times a word occurred in context

        Returns:
            dict[int, int]: context-as-key counts dict
        """
        output = Counter()

        for (_, c), count in self.counts.items():
            output.update({c: count})

        return output

    def get_pair_counts(
        self,
        key_by: Literal["gram", "context"] | None = None,
    ) -> dict[tuple[int, int], int] | dict[int, dict[int, int]]:
        """get multiple key counts dict, the hierarchy depends on `key_by`

        Args:
            key_by (Literal["gram", "context"] | None, optional): None returned flatten tuple level, `gram` returned gram->context levl, `context` returned context->gram levl. Defaults to None.

        Returns:
            dict[tuple[int, int], int] | dict[int, dict[int, int]]: count dict with flatten tuple level key or multilevel key
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
