from glob import glob
import json
from pathlib import Path

import lmdb
import numpy as np


class ImageDatasetConstructorMixin:
    def __init__(self, pattern: str, transform=None, debug_size: int = 0):
        self.path = list(glob(pattern))
        self.transform = transform
        self.debug_size = debug_size

    def __len__(self) -> int:
        return len(self.path) if self.debug_size <= 0 else self.debug_size


class LMDBMixin:
    def post_init(self, lmdb_path: str | Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = lmdb.open(self.lmdb_path.as_posix())
        self.data_size = self.get_data_size()

    def get_data_size(self) -> int:
        with self.env.begin() as tx:
            return int(tx.get(b"count"))

    def __len__(self):
        return self.data_size

    def get_image(self, idx: int) -> np.ndarray:
        with self.env.begin() as tx:
            shape = json.loads(tx.get(f"{idx}_shape".encode("utf-8")))
            img: np.ndarray = np.frombuffer(
                tx.get(f"{idx}".encode("utf-8")), np.uint8
            ).reshape(shape)

        return img
