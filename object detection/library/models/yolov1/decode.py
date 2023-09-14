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
        input_h, input_w = image_size
        stride_x, stride_y = input_w / fm_w, input_h / fm_h
        num_bbox = int((channel_size - self.num_classes) / 5)

        grid_x, grid_y = generate_grid(fm_w, fm_h)

        bbox_info = []
        for b in range(num_bbox):
            # batch_size, 1, grid_size x grid_size
            cx = (
                feature_map[:, (0 + 5 * b) : (1 + 5 * b), :, :] * input_w
                + grid_x * stride_x
            ).reshape(batch_size, 1, -1)
            cy = (
                feature_map[:, (1 + 5 * b) : (2 + 5 * b), :, :] * input_h
                + grid_y * stride_y
            ).reshape(batch_size, 1, -1)
            w = (feature_map[:, (2 + 5 * b) : (3 + 5 * b), :, :] * input_w).reshape(
                batch_size, 1, -1
            )
            h = (feature_map[:, (3 + 5 * b) : (4 + 5 * b), :, :] * input_h).reshape(
                batch_size, 1, -1
            )
            conf = feature_map[:, 4 + 5 * b, :, :].reshape(batch_size, 1, -1)
            prob = feature_map[:, 5 * num_bbox :, :, :].reshape(
                batch_size, self.num_classes, -1
            )
            x = cx - w / 2
            y = cy - h / 2

            bbox_info.append(torch.cat([x, y, w, h, conf, prob], 1))

        result = torch.cat(bbox_info, -1).transpose(1, 2)

        return result
