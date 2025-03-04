import json
from pathlib import Path

import lmdb
import numpy as np


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
