from collections import Counter

import torch

from torchlake.common.helpers.counter import CooccurrenceCounter


class CooccurrenceCounter(CooccurrenceCounter):
    def __init__(
        self, vocab_size: int, neighbor_size: int, padding_idx: int | None = None
    ):
        """word-context co-occurrence counter

        Args:
            vocab_size (int): vocabulary size
            neighbor_size (int): neighbor size
            padding_idx (int | None, optional): index of padding token. Defaults to None.
        """
        super().__init__(vocab_size, padding_idx)
        self.neighbor_size = neighbor_size
        self._offset = vocab_size * neighbor_size
        # counter key is gram * _offset + context

    def update_counts(self, gram: torch.Tensor, context: torch.Tensor):
        """update counts of (word, context)

        Args:
            gram (torch.Tensor): a center word, in shape of (batch*subseq_len, 1)
            context (torch.Tensor): context surround a center word, in shape of (batch*subseq_len, neighbor_size)
        """
        n, neighbor_size = context.shape

        gram = gram.repeat_interleave(neighbor_size, 1).view(-1)
        context = context.view(-1)

        position_encoding = (
            torch.arange(neighbor_size, device=context.device)
            .mul_(self.vocab_size)
            .unsqueeze_(0)
            .repeat(n, 1)
            .view(-1)
        )

        if self.padding_idx is not None:
            not_pad = torch.logical_and(
                gram != self.padding_idx,
                context != self.padding_idx,
            )
            gram, context, position_encoding = (
                gram[not_pad],
                context[not_pad],
                position_encoding[not_pad],
            )

        self.counts.update((gram * self._offset + context + position_encoding).tolist())

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
            dtype=torch.long,
            size=(self.vocab_size, self._offset),
        )
