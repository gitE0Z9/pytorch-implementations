import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class CsvDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        feature_columns: list[str],
        target_column: str,
    ):
        self.paths = paths
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data: list[pd.DataFrame] = []
        self.count = 0

        columns = [*self.feature_columns, *self.target_column]
        self.data = []
        for file_path in self.paths:
            print(f"Reading from {file_path}...")
            df = pd.read_csv(file_path, usecols=columns)
            self.data.append(df)
            self.count += len(df)
        self.data = pd.concat(self.data, ignore_index=True)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> tuple:
        features = self.data.loc[idx, self.feature_columns].values
        target = self.data.loc[idx, self.target_column].values
        return features, target
