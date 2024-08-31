from functools import lru_cache


@lru_cache()
def fnv1a_hash(data: bytes, seed: int = 0x811C9DC5) -> int:
    """Fowler-Noll-Vo hash

    Args:
        data (bytes): Feed-in content
        seed (int): Optional seed value for the hash function (default is 0x811C9DC5)

    Returns:
        int: The integer hash value.
    """
    FNV_prime = 0x01000193
    hash_value = seed

    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_prime) & 0xFFFFFFFF  # Ensure it's a 32-bit hash

    return hash_value
