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
        C = self.context.num_classes
        batch_size, _, _ = decoded.shape
        decoded[:, :, 5:] *= decoded[:, :, 4:5]
        cls_indices = decoded[:, :, 5:].argmax(-1)

        # B, A, # B, A
        sorted_cls_indices, sorted_cls_indices_indices = cls_indices.sort(-1)
        # B, C
        cls_indices_offset = torch.searchsorted(
            sorted_cls_indices,
            torch.arange(C).expand(batch_size, C).contiguous(),
            side="right",
        )
        # for pairwise sequence
        # B, C+1
        cls_indices_offset = torch.cat(
            [
                torch.zeros_like(cls_indices_offset[:, 0:1]),
                cls_indices_offset,
            ],
            1,
        )
        # B, C
        cls_counts = cls_indices_offset.diff(n=1, dim=-1)

        processed_result = []
        for batch_idx in range(batch_size):
            detection_result = []
            for class_index in range(C):
                if cls_counts[batch_idx, class_index] <= 0:
                    continue

                this_class_indices = sorted_cls_indices_indices[
                    batch_idx,
                    cls_indices_offset[batch_idx, class_index] : cls_indices_offset[
                        batch_idx, class_index + 1
                    ],
                ]

                this_class_detection = decoded[batch_idx, this_class_indices]
                best_index = select_best_index(
                    this_class_detection[:, :4],
                    this_class_detection[:, 5 + class_index],
                    postprocess_config,
                )
                detection_result.append(this_class_detection[best_index])
            processed_result.append(torch.cat(detection_result, 0))

        # B x (?, 4+C)
        return processed_result
