import torch

from ..models.yolov1.decode import yolo_postprocess
from .predictor import Predictor


class YOLOPredictor(Predictor):
    def postprocess(
        self,
        output: torch.Tensor,
        img_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        y = super().postprocess(output, img_size)
        y = yolo_postprocess(y, self.context.num_classes, self.inferenceCfg)

        return y
