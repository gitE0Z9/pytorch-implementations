from pydantic import BaseModel


class HardwareCfg(BaseModel):
    DEVICE: str
    NUM_WORKERS: int
    AMP: bool


class ModelCfg(BaseModel):
    NAME: str
    BACKBONE: str
    NUM_ANCHORS: int
    SCALE: int
    CLASSIFIER_PATH: str
    DETECTOR_PATH: str
    ANCHORS_PATH: str


class OptimizerCfg(BaseModel):
    TYPE: str
    LR: float
    DECAY: float
    MOMENTUM: float


class SaveCfg(BaseModel):
    DIR: str
    INTERVAL: int


class TrainingCfg(BaseModel):
    IMAGE_SIZE: int
    BATCH_SIZE: int
    ACC_ITER: int
    OPTIM: OptimizerCfg
    START_EPOCH: int
    END_EPOCH: int
    SAVE: SaveCfg


class DetectorTrainingCfg(TrainingCfg):
    MULTISCALE: bool


class FinetuneCfg(BaseModel):
    EPOCH: int
    BATCH_SIZE: int
    IMAGE_SIZE: int


class ClassifierTrainingCfg(TrainingCfg):
    FINETUNE: FinetuneCfg


class TrainingCfgs(BaseModel):
    DETECTOR: DetectorTrainingCfg
    CLASSIFIER: ClassifierTrainingCfg


class PostProcessParameter(BaseModel):
    CONFLUENCE_THRESH: float
    SIGMA: float
    BETA: float


class InferenceCfg(BaseModel):
    METHOD: str
    CONF_THRESH: float
    NMS_THRESH: float
    PARAMETER: PostProcessParameter


class Setting(BaseModel):
    HARDWARE: HardwareCfg
    MODEL: ModelCfg
    TRAIN: TrainingCfgs
    INFERENCE: InferenceCfg
