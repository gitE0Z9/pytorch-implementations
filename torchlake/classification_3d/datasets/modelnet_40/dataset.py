import urllib.request
from pathlib import Path
from typing import Callable, Literal
import zipfile
import trimesh
from torch.utils.data import Dataset

from .constants import MODELNET40_CLASS_NAMES

URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"


class ModelNet40(Dataset):

    def __init__(
        self,
        root: Path | str,
        mode: Literal["train", "test"] = "train",
        output_type: Literal["mesh", "vertices", "pointcloud"] = "mesh",
        transform: Callable | None = None,
        num_points: int = 1024,
        download: bool = False,
    ):
        self.root = Path(root)
        self.mode = mode
        self.output_type = output_type
        self.transform = transform
        self.num_points = num_points

        if download:
            self.download_data()

        self.filepaths = list(self.root.glob(f"**/{mode}/*.off"))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        filepath: Path = self.filepaths[index]
        data: trimesh.Trimesh = trimesh.load(filepath)
        if self.output_type == "pointcloud":
            data: trimesh.caching.TrackedArray = trimesh.sample.sample_surface_even(
                data, self.num_points
            )[0]
        elif self.output_type == "vertices":
            data: trimesh.caching.TrackedArray = data.vertices

        if self.transform is not None:
            data = self.transform(data)

        label = MODELNET40_CLASS_NAMES.index(filepath.parent.parent.stem)

        return data, label

    def download_data(self):
        zip_path = self.root.joinpath("ModelNet40.zip")
        if not zip_path.exists():
            return

        urllib.request.urlretrieve(URL, zip_path.as_posix())
        with zipfile.ZipFile(zip_path.as_posix(), "r") as f:
            f.extractall(self.root)
