import random
import textwrap

import pytest

from ..utils.tree import build_huffman_tree

random.seed(42)
LARGE_COUNTS = [random.randint(1, 10_000) for _ in range(1_000_000)]


class TestHuffman:
    @pytest.mark.parametrize(
        "counts",
        [
            [5, 2, 3],
            [7, 2, 3],
            [8, 9, 10, 11, 12, 57, 23],
            LARGE_COUNTS,
        ],
    )
    def test_build_tree(self, counts: list[int]):
        """for benchmark and smoke test"""
        build_huffman_tree(counts)

    @pytest.mark.parametrize(
        "counts",
        [
            [5, 2, 3],
            [7, 2, 3],
            [8, 9, 10, 11, 12, 57, 23],
            LARGE_COUNTS,
        ],
    )
    def test_node_count(self, counts: list[int]):
        root = build_huffman_tree(counts)
        assert len(root) == (2 * len(counts) - 1)

    @pytest.mark.parametrize(
        "counts,expected",
        [
            (
                [5, 2, 3],
                textwrap.dedent(
                    """
                    └── 4 (10)
                        ├── 0 (5)
                        └── 3 (5)
                            ├── 1 (2)
                            └── 2 (3)
                    """
                ),
            ),
            (
                [7, 2, 3],
                textwrap.dedent(
                    """
                    └── 4 (12)
                        ├── 3 (5)
                        │   ├── 1 (2)
                        │   └── 2 (3)
                        └── 0 (7)
                    """
                ),
            ),
            (
                [8, 9, 10, 11, 12, 57, 23],
                textwrap.dedent(
                    """
                    └── 12 (130)
                        ├── 5 (57)
                        └── 11 (73)
                            ├── 9 (29)
                            │   ├── 4 (12)
                            │   └── 7 (17)
                            │       ├── 0 (8)
                            │       └── 1 (9)
                            └── 10 (44)
                                ├── 8 (21)
                                │   ├── 2 (10)
                                │   └── 3 (11)
                                └── 6 (23)
                    """
                ),
            ),
        ],
    )
    def test_show_tree(self, counts: list[int], expected: str):
        root = build_huffman_tree(counts)
        assert str(root).strip() == expected.strip()
