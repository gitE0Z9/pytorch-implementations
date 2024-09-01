from pathlib import Path
from unittest import TestCase

from ..datasets import WordAnalogyDataset


class TestWordAnalogyDataset(TestCase):
    def setUp(self) -> None:
        root = Path(__file__).parent
        self.dataset = WordAnalogyDataset(root)

    def tearDown(self) -> None:
        p = self.dataset.path
        if p.exists():
            p.unlink()

    def test_download_data(self):
        assert self.dataset.path.exists()

    def test_load_data(self):
        assert len(self.dataset.data)

    def test_getitem(self):
        item = next(iter(self.dataset))

        self.assertEqual(len(item), 4)
        self.assertIsInstance(item, tuple)
        for ele in item:
            self.assertIsInstance(ele, str)
