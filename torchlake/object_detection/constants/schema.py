from typing import Sequence
from pydantic import BaseModel


class DetectorContext(BaseModel):
    detector_name: str
    dataset: str
    device: str
    num_classes: int
    num_anchors: int | Sequence[int]
    grid_sizes: int | Sequence[int] | None = None
    anchors_path: str = ""
