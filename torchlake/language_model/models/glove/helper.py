from collections import Counter, defaultdict

import torch

from torchlake.common.helpers.counter import CooccurrenceCounter


class CoOccurrenceCounter(CooccurrenceCounter):

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int | None = None,
        enable_distance_weighting: bool = False,
    ):
        """word-context co-occurrence counter

        Args:
            vocab_size (int): size of vocabulary
            padding_idx (int | None, optional): index of padding token. Defaults to None.
            enable_distance_weighting (bool, optional): enable distance weighting on page 7. Defaults to False.
        """
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.enable_distance_weighting = enable_distance_weighting
        self._offset = vocab_size

        # key is gram * vocab_size + context
        if self.enable_distance_weighting:
            self.counts = defaultdict(float)
        else:
            self.counts = Counter()

    def update_counts(self, gram: torch.Tensor, context: torch.Tensor):
        """update counts of (word, context)

        Args:
            gram (torch.Tensor): a center word, in shape of (batch*subseq_len, 1)
            context (torch.Tensor): context surround a center word, in shape of (batch*subseq_len, neighbor_size)
        """
        n, neighbor_size = context.shape
        side_length = neighbor_size // 2
        gram = gram.repeat_interleave(neighbor_size, 1).view(-1)
        context = context.view(-1)

        if self.enable_distance_weighting:
            weights = 1 / torch.arange(1, side_length + 1)
            # neighbor_size => batch*subseq_len, neighbor_size => batch*subseq_len*neighbor_size
            weights = torch.cat((weights.flip(0), weights)).repeat(n, 1).view(-1)

        if self.padding_idx is not None:
            not_pad = torch.logical_and(
                gram != self.padding_idx,
                context != self.padding_idx,
            )
            gram, context = gram[not_pad], context[not_pad]
            if self.enable_distance_weighting:
                weights = weights[not_pad]

        if self.enable_distance_weighting:
            key, inverse = torch.unique(
                gram * self._offset + context,
                return_inverse=True,
            )
            counts = torch.zeros(key.size(0), dtype=weights.dtype).scatter_add_(
                0, inverse, weights
            )

            for i, c in zip(key.tolist(), counts.tolist()):
                self.counts[i] += c
        else:
            self.counts.update((gram * self._offset + context).tolist())

    def get_tensor(self) -> torch.Tensor:
        """get word-context count tensor

        Returns:
            torch.Tensor: word-context count tensor
        """
        row_indices, col_indices, values = [], [], []

        for i, count in self.counts.items():
            row_indices.append(i // self._offset)
            col_indices.append(i % self._offset)
            values.append(count)

        return torch.sparse_coo_tensor(
            [row_indices, col_indices],
            values,
            dtype=torch.float32 if self.enable_distance_weighting else torch.long,
            size=(self.vocab_size, self.vocab_size),
        )
