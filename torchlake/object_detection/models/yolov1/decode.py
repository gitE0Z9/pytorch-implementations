import torch
from torchlake.object_detection.configs.schema import InferenceCfg
from torchlake.object_detection.constants.schema import DetectorContext
from torchlake.object_detection.utils.inference import generate_grid
from torchlake.object_detection.utils.nms import select_best_index


def yolo_postprocess(
    decoded_output: torch.Tensor,
    number_class: int,
    postprocess_config: InferenceCfg,
) -> list[torch.Tensor]:
    """post process yolo decoded output

    Args:
        decoded_output (torch.Tensor): shape (#batch, #anchor * #grid, 5 + #class)
        number_class (int): num class
        postprocess_config (InferenceCfg): post process config for nms

    Returns:
        list[torch.Tensor]: #batch * (#selected, 5 + #class)
    """
    decoded_output[:, :, 5:] *= decoded_output[:, :, 4:5]
    cls_indices = decoded_output[:, :, 5:].argmax(-1)

    processed_result = []
    for decoded, cls_idx in zip(decoded_output, cls_indices):
        detection_result = []
        for class_index in range(number_class):
            is_this_class = cls_idx.eq(class_index)
            if not is_this_class.any():
                continue

            this_class_detection = decoded[is_this_class]
            best_index = select_best_index(
                this_class_detection[:, :4],
                this_class_detection[:, 5 + class_index],
                postprocess_config,
            )
            detection_result.append(this_class_detection[best_index])
        processed_result.append(torch.cat(detection_result, 0))

    return processed_result


class Decoder:
    def __init__(self, context: DetectorContext):
        self.context = context

    def decode(
        self,
        feature_map: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """decode feature map to original size

        Args:
            feature_map (torch.Tensor): shape (#batch, 5 + #class, grid_y, grid_x)
            image_size (tuple[int, int]): (image_y, image_x)

        Returns:
            torch.Tensor: shape (#batch, #anchor * #grid, 5 + #class), in format of (x,y,w,h)
        """
        num_anchors = self.context.num_anchors
        num_classes = self.context.num_classes
        batch_size, _, fm_h, fm_w = feature_map.shape

        # batch_size, boxes, 5, grid_y, grid_x
        feature_map_coord = feature_map[:, : 5 * num_anchors, :, :].unflatten(
            1, (num_anchors, 5)
        )

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
        w = (
            feature_map_coord[:, :, 2, :, :]
            .multiply(input_w)
            .reshape(batch_size, 1, -1)
        )
        h = (
            feature_map_coord[:, :, 3, :, :]
            .multiply(input_h)
            .reshape(batch_size, 1, -1)
        )
        x = cx - w / 2
        y = cy - h / 2

        conf = feature_map_coord[:, :, 4, :, :].reshape(batch_size, 1, -1)
        prob = (
            feature_map[:, 5 * num_anchors :, :, :]
            .unsqueeze(2)
            .repeat(1, 1, num_anchors, 1, 1)
            .reshape(batch_size, num_classes, -1)
        )

        # batch_size, boxes * grid_y * grid_x, 5+C
        result = torch.cat([x, y, w, h, conf, prob], 1).transpose(1, 2)

        return result
