from collections import defaultdict
from pathlib import Path
from typing import Iterator

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from ..constants.enums import PRCurveInterpolation
from ..constants.schema import DetectorContext
from ..utils.eval import average_precision, matched_gt_and_det
from ..utils.train import build_flatten_targets
from .predictor import Predictor


class Evaluator:
    def __init__(self, context: DetectorContext):
        self.context = context

    def set_preprocess(self, *input_size: int):
        """
        Set preprocessing pipeline.
        Be careful, some transformations might drop targets.
        """
        self.preprocess = A.Compose(
            [
                A.Resize(*input_size),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ],
        )

    def evaluate_detector(
        self,
        predictor: Predictor,
        model: nn.Module,
        data: Iterator,
        class_names: list[str],
    ) -> tuple[dict[str, int], pd.DataFrame]:
        """evaluate detector with mAP

        Args:
            predictor (Predictor): predictor class
            model (nn.Module): detector
            data (Iterator): data loader
            class_names (list[str]): class names

        Returns:
            tuple[dict[str, Tuple[List[float], List[float], float]], pd.DataFrame]: precision, recall, AP@0.5
        """
        AP_table = defaultdict(list)

        for imgs, labels in tqdm(data):
            _, _, img_h, img_w = imgs.shape

            imgs: torch.Tensor = imgs.to(self.context.device)
            detections: list[torch.Tensor] = predictor.detect_image(
                model, imgs, is_batch=True
            )

            # recover gt coord in xywh
            labels, spans = build_flatten_targets(labels, delta_coord=False)
            labels[:, 0] = (labels[:, 0] - labels[:, 2] / 2) * img_w
            labels[:, 1] = (labels[:, 1] - labels[:, 3] / 2) * img_h
            labels[:, 2] *= img_w
            labels[:, 3] *= img_h
            labels = labels.split(tuple(spans), 0)

            # shape: num_gt, 5, # shape: num_det, C+5
            for labels_per_image, detections_per_image in zip(labels, detections):
                labels_per_image: torch.Tensor

                cls_prediction = detections_per_image[:, 5:].argmax(1)

                ap_table_per_image = None
                for class_index, class_name in enumerate(class_names):
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
                        this_class_prob = this_class_detection[:, 5 + class_index]

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
                        AP_table[class_name].append(ap_table_per_image)

                    # check gt_num is correct
                    # if has_this_class_groundtruth:
                    #     assert num_gt == ap_table_per_image[2], f"{class_index}"
                    # elif num_gt == 0 and has_this_class_detection:
                    #     assert ap_table_per_image[2] == 0

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
        for class_name in class_names:
            AP[class_name] = average_precision(
                table=AP_table[class_name],
                interpolation=interpolation,
            )

        eval_table = pd.DataFrame(AP, index=["precision", "recall", "AP@0.5"])

        return AP, eval_table

    def run(
        self,
        predictor: Predictor,
        model: nn.Module,
        data: Iterator,
        class_names: list[str],
        verbose: bool = True,
        save_dir: str = None,
        output_filename: str = "eval.csv",
        tf_description: str | None = None,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            predictor (Predictor): predictor class
            model (nn.Module): detector
            data (Iterator): data loader
            class_names (list[str]): class names
            verbose (bool, optional): print mAP table to stdout. Defaults to True.
            save_dir (str, optional): directory to save mAP table. Defaults to None.
            output_filename (str, optional): filename of saved mAP table. Defaults to "eval.csv".
            tf_description (str | None, optional): tensorboard record description. Defaults to None.
        """
        _, eval_table = self.evaluate_detector(predictor, model, data, class_names)

        result_table = eval_table.loc["AP@0.5"].to_frame().T
        result_table.columns = class_names
        result_table["all"] = result_table.mean(axis=None)

        if verbose:
            print(result_table)

        if save_dir:
            p = Path(save_dir)
            p.mkdir(exist_ok=True)
            dst = p.joinpath(output_filename)
            result_table.to_csv(dst.as_posix())

        if tf_description:
            self.writer = SummaryWriter()

            for i, AP in enumerate(result_table["all"]):
                self.writer.add_scalar(
                    "validation mAP",
                    AP,
                    10 * (i + 1),
                    summary_description=tf_description,
                )

            self.writer.close()

        return result_table
