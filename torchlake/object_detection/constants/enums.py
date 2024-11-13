from enum import Enum


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class NetworkStage(Enum):
    INFERENCE = "inference"
    FINETUNE = "finetune"
    SCRATCH = "scratch"


class OperationMode(Enum):
    TRAIN = "train"
    TEST = "test"


class PRCurveInterpolation(Enum):
    ALL = "all"
    VOC = "voc"
    COCO = "coco"
