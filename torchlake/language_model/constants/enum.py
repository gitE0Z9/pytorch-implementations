from enum import Enum


class Word2VecModelType(Enum):
    CBOW = 1
    SKIP_GRAM = 2


class LossType(Enum):
    NEGATIVE_SAMPLING = 1
    HIERARCHICAL_SOFTMAX = 2
    CROSS_ENTROPY = 3


class NgramCombinationMethod(Enum):
    NGRAM_ONLY = 1
    WORD_AND_NGRAM = 2
