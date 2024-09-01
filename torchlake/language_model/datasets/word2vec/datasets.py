from torch.utils.data import Dataset
from pathlib import Path
import urllib.request

URL = {
    "word_analogy": [
        "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt",
        "https://raw.githubusercontent.com/dav/word2vec/master/data/questions-words.txt",
    ],
    "phrases": [
        "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-phrases.txt",
        "https://raw.githubusercontent.com/dav/word2vec/master/data/questions-phrases.txt",
    ],
}


class WordAnalogyDataset(Dataset):

    def __init__(
        self,
        root: str | None = None,
        transform=None,
    ):
        self.name = "WordAnalogy"
        self.path = Path(root / "word-analogy.txt")
        self.transform = transform
        self.data: list[tuple[str, str, str, str]] = self.load_data()

    def download_data(self):
        if not self.path.exists():
            # error flags for each url
            # if both urls failed, show messages
            errors = []

            for url in URL["word_analogy"]:
                request = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(request) as response:
                    if response.status == 200:
                        urllib.request.urlretrieve(url, self.path.as_posix())
                        errors.append(False)
                        break
                    else:
                        errors.append(True)

            assert not all(
                errors
            ), "Failed to download word analogy dataset from sources"

    def load_data(self) -> list[tuple[str, str, str, str]]:
        # check data existence
        self.download_data()

        # parse data
        lines = self.path.read_text().strip().split("\n")
        # drop headers
        lines = lines[1:]
        # make tuple
        lines = [tuple(line.strip().split()) for line in lines]

        if self.transform:
            lines = [self.transform(line) for line in lines]

        return lines

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, str, str, str]:
        return self.data[idx]
