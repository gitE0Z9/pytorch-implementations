import torch
from configs.schema import InferenceCfg
from constants.schema import DetectorContext
from models.ssd.anchor import PriorBox
from utils.nms import select_best_index


class Decoder:
    def __init__(self, context: DetectorContext):
        self.prior = PriorBox()
        self.anchors = self.prior.anchors
        self.context = context

    def decode(
        self,
        prediction: tuple[torch.Tensor, torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        loc_pred, conf_pred = prediction
        input_h, input_w = image_size

        loc_pred = loc_pred * self.anchors
        loc_pred[:, :, 0] *= input_w
        loc_pred[:, :, 1] *= input_h
        loc_pred[:, :, 2] *= input_w
        loc_pred[:, :, 3] *= input_h
        conf_pred = conf_pred.softmax(dim=2)
        decoded = torch.cat([loc_pred, conf_pred], 2)

        return decoded

    def post_process(
        self,
        decoded: torch.Tensor,
        postprocess_config: InferenceCfg,
    ) -> list[torch.Tensor]:
        batch, _, _ = decoded.shape
        cls_index = decoded[:, :, 4:].argmax(2)

        processed_result = []
        for i in range(batch):
            detection_result = []
            for class_index in range(self.context.num_classes):
                is_this_class: torch.Tensor = cls_index[i].eq(class_index)
                if is_this_class.any():
                    this_class_detection = decoded[i, is_this_class]
                    best_index = select_best_index(
                        this_class_detection[:, :4],
                        this_class_detection[:, 5 + class_index],
                        postprocess_config,
                    )
                    detection_result.append(this_class_detection[best_index])
            detection_result = torch.cat(detection_result, 0)
            processed_result.append(detection_result)

        return processed_result
