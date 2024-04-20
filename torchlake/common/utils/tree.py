from __future__ import annotations


class HuffmanNode:
    def __init__(
        self,
        value,
        freq: float | int,
        internal_index: int | None = None,
    ):
        self.value = value
        self.freq = freq
        self.left: HuffmanNode = None
        self.right: HuffmanNode = None
        self.internal_index = internal_index

    def __lt__(self, other: HuffmanNode):
        return self.freq < other.freq

    def __str__(self):
        return f"{self.value},{self.freq}"

    def __len__(self):
        node = self
        count = 1
        if node.left:
            count += len(node.left)
        if node.right:
            count += len(node.right)

        return count
