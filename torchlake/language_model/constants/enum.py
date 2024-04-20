from enum import Enum


class ModelType(Enum):
    CBOW = "CBOW"
    SKIP_GRAM = "SKIP_GRAM"


class LossType(Enum):
    NS = "NS"
    HS = "HS"
    CE = "CE"
