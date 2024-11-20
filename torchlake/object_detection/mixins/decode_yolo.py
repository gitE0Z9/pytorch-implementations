import torch
from torchlake.object_detection.configs.schema import InferenceCfg
from torchlake.object_detection.utils.nms import select_best_index


class YOLODecodeMixin:
    def post_process(
        self,
        decoded: torch.Tensor,
        postprocess_config: InferenceCfg,
    ) -> list[torch.Tensor]:
        """post process yolo decoded output

        Args:
            decoded (torch.Tensor): shape (#batch, #anchor * #grid, 5 + #class)
            postprocess_config (InferenceCfg): post process config for nms

        Returns:
            list[torch.Tensor]: #batch * (#selected, 5 + #class)
        """
        decoded[:, :, 5:] *= decoded[:, :, 4:5]
        cls_indices = decoded[:, :, 5:].argmax(-1)

        processed_result = []
        for decoded, cls_idx in zip(decoded, cls_indices):
            detection_result = []
            for class_index in range(self.context.num_classes):
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
