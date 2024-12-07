from pathlib import Path

import pandas as pd
from typing import Callable
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CSVDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        feature_columns: list[str],
        target_column: str,
        transform: Callable | None = ToTensor(),
        label_transform: Callable | None = ToTensor(),
    ):
        self.path = path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.transform = transform
        self.label_transform = label_transform

        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(
            self.path,
            usecols=[*self.feature_columns, *self.target_column],
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        features = self.data.iloc[idx, self.feature_columns].values
        target = self.data.iloc[idx, self.target_column].values

        if self.transform is not None:
            features = self.transform(features)

        if self.label_transform is not None:
            target = self.label_transform(target)

        return features, target
