import torch
from utils.inference import generate_grid
from utils.nms import select_best_index
from constants.schema import DetectorContext
from configs.schema import InferenceCfg


def yolo_postprocess(
    decoded_output: torch.Tensor,
    number_class: int,
    postprocess_config: InferenceCfg,
) -> list[torch.Tensor]:
    batch, _, _ = decoded_output.shape
    decoded_output[:, :, 5:] = decoded_output[:, :, 4:5] * decoded_output[:, :, 5:]
    cls_index = decoded_output[:, :, 5:].argmax(2)

    processed_result = []
    for i in range(batch):
        detection_result = []
        for class_index in range(number_class):
            is_this_class = cls_index[i].eq(class_index)
            if is_this_class.any():
                this_class_detection = decoded_output[i, is_this_class]
                best_index = select_best_index(
                    this_class_detection[:, :4],
                    this_class_detection[:, 5 + class_index],
                    postprocess_config,
                )
                detection_result.append(this_class_detection[best_index])
        detection_result = torch.cat(detection_result, 0)
        processed_result.append(detection_result)

    return processed_result


class Decoder:
    def __init__(self, context: DetectorContext):
        self.num_classes = context.num_classes

    def decode(
        self,
        feature_map: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, channel_size, fm_h, fm_w = feature_map.shape
        num_bbox = int((channel_size - self.num_classes) / 5)
        feature_map = feature_map.view(batch_size, num_bbox, 5, fm_h, fm_w)

        input_h, input_w = image_size
        stride_x, stride_y = input_w / fm_w, input_h / fm_h

        grid_x, grid_y = generate_grid(fm_w, fm_h)

        # batch_size, 1, boxes * grid_y * grid_x
        cx = (
            feature_map[:, :, 0:1, :, :]
            .multiply(input_w)
            .add(grid_x * stride_x)
            .view(batch_size, 1, -1)
        )
        cy = (
            feature_map[:, :, 1:2, :, :]
            .multiply(input_h)
            .add(grid_y * stride_y)
            .view(batch_size, 1, -1)
        )

        w = feature_map[:, :, 2:3, :, :].multiply(input_w).view(batch_size, 1, -1)
        h = feature_map[:, :, 3:4, :, :].multiply(input_h).view(batch_size, 1, -1)
        conf = feature_map[:, :, 4:5, :, :].view(batch_size, 1, -1)
        prob = feature_map[:, :, 5 * num_bbox :, :, :].view(
            batch_size, self.num_classes, -1
        )
        x = cx - w / 2
        y = cy - h / 2

        # batch_size, boxes * grid_y * grid_x, 5+C
        result = torch.cat([x, y, w, h, conf, prob], 1).transpose(1, 2)

        return result
