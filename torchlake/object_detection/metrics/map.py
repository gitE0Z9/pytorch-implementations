from collections import defaultdict
from typing import Sequence

import pandas as pd
import torch

from ..constants.enums import PRCurveInterpolation
from ..constants.schema import DetectorContext
from ..utils.eval import average_precision, matched_gt_and_det


class MeanAveragePrecision:
    def __init__(self, context: DetectorContext, class_names: Sequence[str]):
        self.context = context
        self.class_names = class_names
        self.ap_table = defaultdict(list)

    def update(self, detections: list[torch.Tensor], labels: list[torch.Tensor]):
        for labels_per_image, detections_per_image in zip(labels, detections):
            if "yolo" in self.context.detector_name:
                cls_prediction = detections_per_image[:, 5:].argmax(1)
            else:
                cls_prediction = detections_per_image[:, 4:].argmax(1)

            ap_table_per_image = None
            for class_index, class_name in enumerate(self.class_names):
                if "yolo" not in self.context.detector_name:
                    class_index += 1

                this_class_detection = detections_per_image[
                    cls_prediction.eq(class_index)
                ]
                num_det = this_class_detection.size(0)
                # have det
                has_this_class_detection = num_det != 0

                this_class_label = labels_per_image[
                    labels_per_image[:, -1] == class_index
                ]
                num_gt = this_class_label.size(0)
                # have gt
                has_this_class_groundtruth = num_gt != 0

                # if detected
                if has_this_class_detection:
                    if "yolo" in self.context.detector_name:
                        this_class_prob = this_class_detection[:, 5 + class_index]
                    else:
                        this_class_prob = this_class_detection[:, 4 + class_index]

                    # has gt and det
                    if has_this_class_groundtruth:
                        this_class_detection = torch.cat(
                            [
                                this_class_detection[:, :4],
                                this_class_prob.view(-1, 1),
                            ],
                            dim=1,
                        )
                        ap_table_per_image = matched_gt_and_det(
                            this_class_detection,
                            this_class_label[:, :4],
                        )
                    # have no gt but det => FP
                    else:
                        ap_table_per_image = (
                            torch.zeros(num_det),
                            this_class_prob,
                            0,
                        )

                # no detection
                else:
                    # have gt but no det => FN
                    if has_this_class_groundtruth:
                        ap_table_per_image = (None, None, num_gt)
                    # no gt or det => pass
                    # else:
                    #     ap_table_per_image = (None, None, 0)

                if ap_table_per_image is not None:
                    self.ap_table[class_name].append(ap_table_per_image)

                # check gt_num is correct
                # if has_this_class_groundtruth:
                #     assert num_gt == ap_table_per_image[2], f"{class_index}"
                # elif num_gt == 0 and has_this_class_detection:
                #     assert ap_table_per_image[2] == 0

    def score(self):
        # calculate pr curve and auc
        interpolation_mapping = {
            "VOC": PRCurveInterpolation.VOC.value,
            "COCO": PRCurveInterpolation.COCO.value,
        }

        interpolation = interpolation_mapping.get(
            self.context.dataset,
            PRCurveInterpolation.ALL.value,
        )

        AP = {}
        for class_name in self.class_names:
            AP[class_name] = average_precision(
                table=self.ap_table[class_name],
                interpolation=interpolation,
            )

        return pd.DataFrame(AP, index=["precision", "recall", "AP@0.5"])
