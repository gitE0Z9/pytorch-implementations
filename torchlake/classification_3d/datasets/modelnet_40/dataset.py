import json
import urllib.request
from pathlib import Path
from typing import Callable, Literal
import zipfile
import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import trimesh
except:
    pass

try:
    import open3d as o3d
except:
    pass

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
        backend: Literal["trimesh", "open3d"] = "open3d",
    ):
        self.root = Path(root)
        self.mode = mode
        self.output_type = output_type
        self.transform = transform
        self.num_points = num_points
        self.backend = backend

        if download:
            self.download_data()

        self.filepaths = list(self.root.glob(f"**/{mode}/*.off"))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        filepath: Path = self.filepaths[index]

        data = self.get_data(filepath)
        label = self.get_label(filepath)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def get_data(self, filepath: Path):
        return {
            "trimesh": self.get_data_by_trimesh,
            "open3d": self.get_data_by_open3d,
        }[self.backend](filepath)

    def get_data_by_trimesh(self, filepath: Path):
        data: trimesh.Trimesh = trimesh.load(filepath)
        if self.output_type == "pointcloud":
            data: trimesh.caching.TrackedArray = trimesh.sample.sample_surface(
                data, self.num_points
            )[0]
        elif self.output_type == "vertices":
            data: trimesh.caching.TrackedArray = data.vertices

        return data

    def get_data_by_open3d(self, filepath: Path):
        data = o3d.io.read_triangle_mesh(filepath)
        if self.output_type == "pointcloud":
            data = data.sample_points_uniformly(self.num_points)
            data = np.asarray(data.points)
        elif self.output_type == "vertices":
            data = np.asarray(data.vertices)

        return data

    def get_label(self, filepath: Path) -> int:
        return MODELNET40_CLASS_NAMES.index(filepath.parent.parent.stem)

    def download_data(self):
        zip_path = self.root.joinpath("ModelNet40.zip")
        if not zip_path.exists():
            return

        urllib.request.urlretrieve(URL, zip_path.as_posix())
        with zipfile.ZipFile(zip_path.as_posix(), "r") as f:
            f.extractall(self.root)

    def to_lmdb(self, env: lmdb.Environment):
        assert self.output_type in [
            "pointcloud",
            "vertices",
        ], "Only point cloud and vertices output supported."

        with env.begin(write=True) as tx:
            for i, (data, label) in enumerate(tqdm(self)):
                tx.put(f"{i}".encode("utf-8"), data.tobytes())
                tx.put(
                    f"{i}_shape".encode("utf-8"), str(list(data.shape)).encode("utf-8")
                )
                tx.put(f"{i}_label".encode("utf-8"), str(label).encode("utf-8"))

            tx.put(b"count", str(len(self)).encode("utf-8"))


class ModelNet40FromLMDB(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        transform=None,
        num_points: int = 1024,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.transform = transform
        self.num_points = num_points

        self.env = lmdb.open(lmdb_path)

        self.data_size = self.get_data_size()

    def get_data_size(self) -> int:
        with self.env.begin() as tx:
            return int(tx.get(b"count"))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if idx >= self.data_size:
            raise IndexError(f"invalid index {idx}")

        data = self._get_img(idx)
        label = self._get_label(idx)

        # uniformly sampled from dense points
        data = data[np.random.permutation(len(data))[: self.num_points]]

        if self.transform:
            data = self.transform(data)

        return data, label

    def _get_img(self, idx: int) -> np.ndarray:
        with self.env.begin() as tx:
            shape = json.loads(tx.get(f"{idx}_shape".encode("utf-8")))
            data: np.ndarray = np.frombuffer(
                tx.get(f"{idx}".encode("utf-8")), np.float32
            ).reshape(shape)

        return data

    def _get_label(self, idx: int) -> tuple[list, str]:
        with self.env.begin() as tx:
            label = int(tx.get(f"{idx}_label".encode("utf-8")))

        return label
