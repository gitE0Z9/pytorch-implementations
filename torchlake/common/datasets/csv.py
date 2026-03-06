from pathlib import Path

import polars as pl
from typing import Any, Callable
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        feature_columns: list[str] = [],
        label_columns: list[str] = [],
        transform: Callable[[pl.DataFrame], Any] | None = None,
        label_transform: Callable[[pl.DataFrame], Any] | None = None,
    ):
        self.path = path
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.transform = transform
        self.label_transform = label_transform

        self.load_data()

    def load_data(self):
        kwargs = {}
        if self.feature_columns and self.label_columns:
            kwargs["columns"] = self.feature_columns + self.label_columns
        else:
            columns = pl.read_csv(self.path, n_rows=0).columns
            raise ValueError(
                f"please assign feature_columns and target_columns first. available options are {columns}"
            )

        df = pl.read_csv(self.path, **kwargs)
        self.features = df.select(self.feature_columns)
        self.labels = df.select(self.label_columns)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.label_transform is not None:
            y = self.label_transform(y)

        return x, y
