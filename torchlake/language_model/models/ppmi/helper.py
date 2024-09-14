from collections import Counter, defaultdict
from typing import Literal

import torch


class CoOccurrenceCounter:

    def __init__(self):
        """word context co-occurrence matrix"""
        super(CoOccurrenceCounter, self).__init__()
        self.counts: dict[tuple[str, tuple[str]], int] = Counter()

    def update_counts(self, gram: torch.Tensor, context: torch.Tensor):
        """update counts of (word, context)

        Args:
            gram (torch.Tensor): a center word
            context (torch.Tensor): context surround a center word
        """
        self.counts.update(
            zip(
                gram.squeeze().tolist(),
                map(lambda ele: tuple(ele), context.tolist()),
            ),
        )

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
