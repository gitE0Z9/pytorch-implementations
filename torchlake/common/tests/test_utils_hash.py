import os

import pytest

from ..utils.hash import fnv1a_hash


def generate_random_bytes(size: int) -> bytes:
    """Generate random bytes of specified size."""
    return os.urandom(size)


@pytest.mark.parametrize("size", [10**p for p in range(2, 8)])
def test_fnv1a_hash_large_input(size: int):
    x = generate_random_bytes(size)

    assert isinstance(fnv1a_hash(x), int)
