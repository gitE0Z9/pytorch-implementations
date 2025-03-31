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

    @pytest.mark.parametrize(
        "counts,expected",
        [
            ([5, 2, 3], "4:10,0:5,3:5,1:2,2:3"),
            ([7, 2, 3], "4:12,0:7,3:5,1:2,2:3"),
            (
                [8, 9, 10, 11, 12, 57, 23],
                "12:130,5:57,11:73,9:29,4:12,10:44,6:23,7:17,0:8,1:9,8:21,2:10,3:11",
            ),
        ],
    )
    def test_build_tree(self, counts: list[int], expected: str):
        root = build_huffman_tree(counts)
        assert root.show() == expected
