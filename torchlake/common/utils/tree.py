from __future__ import annotations

from collections import deque
import heapq


class HuffmanNode:
    def __init__(
        self,
        value,
        freq: float | int,
    ):
        self.value = value
        self.freq = freq
        self.left: HuffmanNode | None = None
        self.right: HuffmanNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other: HuffmanNode):
        return self.freq < other.freq

    def __str__(self):
        return f"{self.value},{self.freq}"

    def __len__(self):
        count = 0
        queue = deque([self])
        while queue:
            node = queue.popleft()
            count += 1

            if node.left is not None:
                if node.left.is_leaf:
                    count += 1
                if not node.left.is_leaf:
                    queue.append(node.left)

            if node.right is not None:
                if node.right.is_leaf:
                    count += 1
                if not node.right.is_leaf:
                    queue.append(node.right)

        return count

    def show(self):
        output = []
        queue = deque([self])
        while queue:
            node = queue.popleft()
            output.append(f"{node.value}:{node.freq}")

            if node.left is not None:
                if node.left.is_leaf:
                    output.append(f"{node.left.value}:{node.left.freq}")
                if not node.left.is_leaf:
                    queue.append(node.left)

            if node.right is not None:
                if node.right.is_leaf:
                    output.append(f"{node.right.value}:{node.right.freq}")
                if not node.right.is_leaf:
                    queue.append(node.right)

        return ",".join(output)


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
        internal_index += 1
        heapq.heappush(nodes, combined_node)

    return nodes.pop(0)
