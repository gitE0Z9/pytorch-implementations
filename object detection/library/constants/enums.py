from enum import Enum


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class NetworkType(Enum):
    CLASSIFIER = "CLASSIFIER"
    DETECTOR = "DETECTOR"


class NetworkStage(Enum):
    INFERENCE = "inference"
    FINETUNE = "finetune"
    SCRATCH = "scratch"


class PRCurveInterpolation(Enum):
    ALL = "all"
    VOC = "voc"
    COCO = "coco"
