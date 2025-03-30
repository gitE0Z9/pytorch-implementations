import pytest

from ..utils.tree import build_huffman_tree


class TestHuffman:
    @pytest.mark.parametrize(
        "counts",
        [
            [5, 2, 3],
            [7, 2, 3],
            [8, 9, 10, 11, 12, 57, 23],
        ],
    )
    def test_node_count(self, counts: list[int]):
        root = build_huffman_tree(counts)
        assert len(root) == (2 * len(counts) - 1)
