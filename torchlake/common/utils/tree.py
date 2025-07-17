from __future__ import annotations

import heapq


class HuffmanNode:
    __slot__ = ("value", "freq", "left", "right", "_size")

    def __init__(
        self,
        value,
        freq: float | int,
    ):
        self.value = value
        self.freq = freq
        self.left: HuffmanNode | None = None
        self.right: HuffmanNode | None = None
        self._size = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other: HuffmanNode):
        return self.freq < other.freq

    def __str__(self):
        return f"{self.value},{self.freq}"

    def __len__(self):
        if self._size > 0:
            return self._size

        self._size = 1
        if not self.is_leaf:
            if self.left is not None:
                self._size += len(self.left)
            if self.right is not None:
                self._size += len(self.right)

        return self._size

    def __str__(self, prefix: str = "", is_left: bool = False) -> str:
        if self is None:
            return

        connector = "├── " if is_left else "└── "
        lines = [prefix + connector + f"{self.value} ({self.freq})"]

        if self.left:
            lines.append(
                self.left.__str__(prefix + ("│   " if is_left else "    "), True)
            )
        if self.right:
            lines.append(
                self.right.__str__(prefix + ("│   " if is_left else "    "), False)
            )

        return "\n".join(lines)


def build_huffman_tree(freqs: list[int | float]) -> HuffmanNode:
    """Build huffman tree with frequencies.

    Returns:
        HuffmanNode: root of huffman tree
    """
    N = len(freqs)
    nodes: list[HuffmanNode] = [
        HuffmanNode(index, count) for index, count in enumerate(freqs)
    ]
    heapq.heapify(nodes)

    # index for combined node, start from lowest one
    internal_index = 0
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        combined_node = HuffmanNode(N + internal_index, left.freq + right.freq)
        combined_node.left = left
        combined_node.right = right
        combined_node._size = len(left) + len(right) + 1
        internal_index += 1
        heapq.heappush(nodes, combined_node)

    return nodes.pop(0)
