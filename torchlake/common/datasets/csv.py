from pathlib import Path

import pandas as pd
from typing import Callable
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        feature_columns: list[str],
        target_columns: list[str],
        transform: Callable | None = None,
        label_transform: Callable | None = None,
    ):
        self.path = path
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.transform = transform
        self.label_transform = label_transform

        self.load_data()

    def load_data(self):
        kwargs = {}
        if self.feature_columns or self.target_columns:
            kwargs["usecols"] = self.feature_columns + self.target_columns

        self.data = pd.read_csv(self.path, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        features = self.data.iloc[idx][self.feature_columns].tolist()[0]
        target = self.data.iloc[idx][self.target_columns].tolist()[0]

        if self.transform is not None:
            features = self.transform(features)

        if self.label_transform is not None:
            target = self.label_transform(target)

        return features, target
