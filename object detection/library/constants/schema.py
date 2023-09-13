from pydantic import BaseModel


class DetectorContext(BaseModel):
    detector_name: str
    dataset: str
    device: str
    num_classes: int
    num_anchors: int
