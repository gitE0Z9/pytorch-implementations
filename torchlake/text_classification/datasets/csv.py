import pandas as pd
from torch.utils.data import Dataset


class CsvDataset(Dataset):
    def __init__(self, path: str | list[str]):
        self.path = path

        for path in self.path:
            print("reading from", path, "...")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        ...
