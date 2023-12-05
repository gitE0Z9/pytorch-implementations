import torch
from object_detection.configs.schema import InferenceCfg
from object_detection.constants.schema import DetectorContext
from object_detection.utils.inference import generate_grid
from object_detection.utils.nms import select_best_index


def yolo_postprocess(
    decoded_output: torch.Tensor,
    number_class: int,
    postprocess_config: InferenceCfg,
) -> list[torch.Tensor]:
    batch, _, _ = decoded_output.shape
    decoded_output[:, :, 5:] = decoded_output[:, :, 4:5] * decoded_output[:, :, 5:]
    print(decoded_output[:, :, 5:].round(decimals=2).max(2))
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
        feature_map = feature_map.unsqueeze(1)
        batch_size, _, channel_size, fm_h, fm_w = feature_map.shape
        num_bbox = int((channel_size - self.num_classes) / 5)
        feature_map_coord = feature_map[:, :, : 5 * num_bbox, :, :].reshape(
            batch_size, num_bbox, 5, fm_h, fm_w
        )  # N, 2, 5, 7, 7

        input_h, input_w = image_size
        stride_x, stride_y = input_w / fm_w, input_h / fm_h

        grid_x, grid_y = generate_grid(fm_w, fm_h)

        # batch_size, 1, boxes * grid_y * grid_x
        cx = (
            feature_map_coord[:, :, 0, :, :]
            .multiply(input_w)
            .add(grid_x * stride_x)
            .reshape(batch_size, 1, -1)
        )
        cy = (
            feature_map_coord[:, :, 1, :, :]
            .multiply(input_h)
            .add(grid_y * stride_y)
            .reshape(batch_size, 1, -1)
        )

        w = feature_map_coord[:, :, 2, :, :].multiply(input_w).reshape(batch_size, 1, -1)
        h = feature_map_coord[:, :, 3, :, :].multiply(input_h).reshape(batch_size, 1, -1)
        conf = feature_map_coord[:, :, 4, :, :].reshape(batch_size, 1, -1)
        prob = (
            feature_map[:, :, 5 * num_bbox :, :, :]
            .repeat(1, num_bbox, 1, 1, 1)
            .reshape(batch_size, self.num_classes, -1)
        )
        x = cx - w / 2
        y = cy - h / 2

        # batch_size, boxes * grid_y * grid_x, 5+C
        result = torch.cat([x, y, w, h, conf, prob], 1).transpose(1, 2)

        return result
