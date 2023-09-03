import torch

from typing import Tuple

from utils.nms import select_best_index


def model_predict(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if img.dim() == 3:
            img = img.unsqueeze(0)
        pred = model(img)

        pred = pred.detach().cpu()
        img = img.detach().cpu()

    return pred


def decode_model_prediction(
    model_name: str,
    output: torch.Tensor,
    img_h: int,
    img_w: int,
    **kwargs,
):
    mapping = {
        "yolov1": yolov1_decode,
        "yolov2": yolov2_decode,
    }

    return mapping.get(model_name)(
        output,
        (img_h, img_w),
        **kwargs,
    )


def generate_grid(grid_x: int, grid_y: int) -> torch.Tensor:
    y_offset, x_offset = torch.meshgrid(
        torch.arange(grid_x),
        torch.arange(grid_y),
        indexing="xy",
    )

    return x_offset, y_offset


def yolov1_decode(
    feature_map: torch.Tensor,
    image_size: Tuple[int, int],
    num_classes: int,
) -> torch.Tensor:
    batch_size, channel_size, fm_h, fm_w = feature_map.shape
    input_h, input_w = image_size
    stride_x, stride_y = input_w / fm_w, input_h / fm_h
    num_bbox = int((channel_size - num_classes) / 5)

    grid_x, grid_y = generate_grid(fm_w, fm_h)

    bbox_info = []
    for b in range(num_bbox):
        # batch_size, 1, grid_size x grid_size
        cx = (
            feature_map[:, (1 + 5 * b) : (2 + 5 * b), :, :] * input_w
            + grid_x * stride_x
        ).reshape(batch_size, 1, -1)
        cy = (
            feature_map[:, (2 + 5 * b) : (3 + 5 * b), :, :] * input_h
            + grid_y * stride_y
        ).reshape(batch_size, 1, -1)
        w = (feature_map[:, (3 + 5 * b) : (4 + 5 * b), :, :] * input_w).reshape(
            batch_size, 1, -1
        )
        h = (feature_map[:, (4 + 5 * b) : (5 + 5 * b), :, :] * input_h).reshape(
            batch_size, 1, -1
        )
        conf = feature_map[:, 0 + 5 * b, :, :].reshape(batch_size, 1, -1)
        prob = feature_map[:, 5 * num_bbox :, :, :].reshape(batch_size, num_classes, -1)
        x = cx - w / 2
        y = cy - h / 2

        bbox_info.append(torch.cat([x, y, w, h, conf, prob], 1))

    result = torch.cat(bbox_info, -1).transpose(1, 2)

    return result


def yolov2_decode(
    feature_map: torch.Tensor,
    image_size: Tuple[int, int],
    anchors: torch.Tensor,
) -> torch.Tensor:
    batch_size, channel_size, fm_h, fm_w = feature_map.shape
    input_h, input_w = image_size
    stride_x, stride_y = input_w / fm_w, input_h / fm_h
    num_anchors = anchors.size(1)
    num_classes = (channel_size // num_anchors) - 5
    feature_map = feature_map.reshape(
        batch_size, num_anchors, channel_size // num_anchors, fm_h, fm_w
    )

    grid_x, grid_y = generate_grid(fm_w, fm_h)

    bbox_info = []
    for a in range(num_anchors):
        # batch_size, 1, grid_size x grid_size
        cx = (
            feature_map[:, a, 0:1, :, :].sigmoid() * input_w + grid_x * stride_x
        ).reshape(batch_size, 1, -1)
        cy = (
            feature_map[:, a, 1:2, :, :].sigmoid() * input_h + grid_y * stride_y
        ).reshape(batch_size, 1, -1)
        w = (
            feature_map[:, a, 2:3, :, :].exp() * anchors[:, a, 0:1, :, :] * input_w
        ).reshape(batch_size, 1, -1)
        h = (
            feature_map[:, a, 3:4, :, :].exp() * anchors[:, a, 1:2, :, :] * input_h
        ).reshape(batch_size, 1, -1)
        conf = feature_map[:, a, 4:5, :, :].sigmoid().reshape(batch_size, 1, -1)
        prob = (
            feature_map[:, a, 5:, :, :].softmax(1).reshape(batch_size, num_classes, -1)
        )
        x = cx - w / 2
        y = cy - h / 2

        bbox_info.append(torch.cat([x, y, w, h, conf, prob], 1))

    result = torch.cat(bbox_info, -1).transpose(
        1, 2
    )  # batch_size, grid_size * grid_size * num_anchors, num_classes + 4

    return result


def yolo_postprocess(
    decoded_output: torch.Tensor,
    number_class: int,
    postprocess_config: dict,
) -> torch.Tensor:
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

    # print(processed_result)

    return processed_result
