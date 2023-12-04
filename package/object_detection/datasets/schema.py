from pydantic import BaseModel


class DatasetCfg(BaseModel):
    ROOT: str
    CSV_ROOT: str | None = None
    CLASSES_PATH: str
    NUM_CLASSES: int
