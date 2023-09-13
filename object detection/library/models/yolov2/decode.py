import torch
from constants.schema import DetectorContext
from models.yolov2.anchor import PriorBox
from utils.inference import generate_grid


class Decoder:
    def __init__(self, context: DetectorContext):
        self.anchors = PriorBox(context.num_anchors, context.dataset).anchors
        self.anchors.to(context.device)

    def decode(
        self,
        feature_map: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, channel_size, fm_h, fm_w = feature_map.shape
        input_h, input_w = image_size
        stride_x, stride_y = input_w / fm_w, input_h / fm_h
        num_anchors = self.anchors.size(1)
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
                feature_map[:, a, 2:3, :, :].exp()
                * self.anchors[:, a, 0:1, :, :]
                * input_w
            ).reshape(batch_size, 1, -1)
            h = (
                feature_map[:, a, 3:4, :, :].exp()
                * self.anchors[:, a, 1:2, :, :]
                * input_h
            ).reshape(batch_size, 1, -1)
            conf = feature_map[:, a, 4:5, :, :].sigmoid().reshape(batch_size, 1, -1)
            prob = (
                feature_map[:, a, 5:, :, :]
                .softmax(1)
                .reshape(batch_size, num_classes, -1)
            )
            x = cx - w / 2
            y = cy - h / 2

            bbox_info.append(torch.cat([x, y, w, h, conf, prob], 1))

        # batch_size, grid_size * grid_size * num_anchors, num_classes + 4
        result = torch.cat(bbox_info, -1).transpose(1, 2)

        return result
