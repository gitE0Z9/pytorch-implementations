from typing import List

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from constants.enums import (
    NetworkType,
    OperationMode,
    PRCurveInterpolation,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.eval import average_precision, matched_gt_and_det
from utils.inference import (
    generate_grid,
    model_predict,
)
from utils.train import build_targets

from controller.controller import Controller


class Evaluator(Controller):
    def set_preprocess(self, input_size: int):
        """
        Set preprocessing pipeline.
        Be careful, some transformations might drop targets.
        """
        preprocess = A.Compose(
            [
                A.Resize(input_size, input_size),
                A.Normalize(),
                ToTensorV2(),
            ],
        )

        self.data[OperationMode.TEST.value]["preprocess"] = preprocess

    def evaluate_detector(self):
        """evaluate mAP for YOLO"""
        training_cfg = self.get_training_cfg()
        dataset_cfg = self.get_dataset_cfg()

        IMAGE_SIZE = training_cfg.IMAGE_SIZE
        SCALE = self.cfg.MODEL.SCALE
        GRID_SIZE = IMAGE_SIZE // SCALE
        NUM_CLASSES = dataset_cfg.NUM_CLASSES

        AP = {c: 0 for c in self.class_names}
        AP_table = {c: [] for c in self.class_names}

        for imgs, labels in tqdm(self.data[OperationMode.TEST.value]["loader"]):
            imgs = imgs.to(self.device)
            batch_size, _, img_h, img_w = imgs.shape
            output = model_predict(self.model, imgs)
            detections = self.postprocess(output, (img_h, img_w))

            # recover gt coord in xywh
            # N, 1, 25, 13, 13
            labels = build_targets(
                labels,
                (batch_size, 1, NUM_CLASSES + 5, GRID_SIZE, GRID_SIZE),
            )

            # return w,h
            labels[:, :, 2:4] = labels[:, :, 2:4] * IMAGE_SIZE
            grid_x, grid_y = generate_grid(GRID_SIZE, GRID_SIZE)
            # return x1
            labels[:, :, 0] = (
                labels[:, :, 0] * IMAGE_SIZE + grid_x * SCALE - labels[:, :, 2] / 2
            )

            # return y1
            labels[:, :, 1] = (
                labels[:, :, 1] * IMAGE_SIZE + grid_y * SCALE - labels[:, :, 3] / 2
            )

            for i in range(batch_size):
                # 1, 25, 13, 13
                single_label = labels[i]
                # 169, 25
                single_label = single_label.view(NUM_CLASSES + 5, -1).transpose(0, 1)
                # leave exist object
                single_label = single_label[single_label[:, 4] == 1]
                label_count = single_label[:, 5:].sum(0)

                single_detection = detections[i]  # N, C+5
                cls_prediction = single_detection[:, 5:].argmax(1)

                for class_index, class_name in enumerate(self.class_names):
                    this_class_detection = single_detection[
                        cls_prediction.eq(class_index)
                    ]
                    has_this_class_detection = this_class_detection.size(0) != 0

                    this_class_label = single_label[
                        single_label[:, 5 + class_index] == 1, :4
                    ]
                    # have gt and det
                    has_this_class_groundtruth = this_class_label.size(0) != 0

                    # if detected
                    if has_this_class_detection:
                        this_class_prob = this_class_detection[:, 5 + class_index]

                        # has gt and det
                        if has_this_class_groundtruth:
                            this_class_detection = torch.cat(
                                [
                                    this_class_detection[:, :4],
                                    this_class_prob.reshape(-1, 1),
                                ],
                                dim=1,
                            )
                            tmp_table = matched_gt_and_det(
                                this_class_detection,
                                this_class_label,
                            )
                        # have no gt but det
                        else:
                            tmp_table = (
                                torch.zeros(this_class_detection.size(0)),
                                this_class_prob,
                                0,
                            )

                    # no detection
                    else:
                        # have gt but no det
                        if has_this_class_groundtruth:
                            tmp_table = (None, None, label_count[class_index].item())
                        # no gt or det
                        else:
                            tmp_table = (None, None, 0)

                    AP_table[class_name].append(tmp_table)  # tuple

                    # check gt_num is correct
                    # if has_this_class_groundtruth:
                    #     assert label_count[class_index] == tmp_table[2], f'{i, class_index}'
                    # elif label_count[class_index]==0 and has_this_class_detection:
                    #     assert tmp_table[2] == 0

        # calculate pr curve and auc
        if self.dataset_name == "VOC":
            interpolation = PRCurveInterpolation.VOC.value
        elif self.dataset_name == "COCO":
            interpolation = PRCurveInterpolation.COCO.value
        else:
            interpolation = PRCurveInterpolation.ALL.value

        for class_index in AP:
            AP[class_index] = average_precision(
                table=AP_table[class_index],
                interpolation=interpolation,
            )

        eval_table = pd.DataFrame(AP, index=["precision", "recall", "AP@0.5"])

        return AP, eval_table

    def evaluate_classifier(self):
        """evaluate accuracy for classifier"""
        accuracy = 0
        for imgs, labels in tqdm(self.data[OperationMode.TEST.value]["loader"]):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            predictions = self.model(imgs).argmax(1)
            accuracy += predictions.eq(labels).sum().item()
        print(
            "validation accuracy",
            accuracy / len(self.data[OperationMode.TEST.value]["dataset"]),
        )

    def evaluate(self, weight_paths: List[str], store: bool, description: str):
        """main function for evaluate"""
        self.prepare_inference()
        self.load_dataset(OperationMode.TEST.value)
        self.model.eval()

        if self.network_type == NetworkType.DETECTOR.value:
            results = {}

            for weight_path in weight_paths:
                print(weight_path)
                self.load_weight(weight_path)
                _, eval_table = self.evaluate_detector()
                results[weight_path] = eval_table.loc["AP@0.5"]

            cmap = [list(APs) + [APs.mean()] for _, APs in results.items()]
            result_table = pd.DataFrame(
                cmap,
                columns=self.class_names + ["all"],
                index=results.keys(),
            )

            print(result_table)

            if store:
                result_table.to_csv("eval.csv")

            if description:
                self.writer = SummaryWriter()

                for i, AP in enumerate(result_table["all"]):
                    self.writer.add_scalar(
                        "validation mAP",
                        AP,
                        10 * (i + 1),
                        summary_description=description,
                    )

                self.writer.close()

        elif self.network_type == NetworkType.CLASSIFIER.value:
            for weight_path in weight_paths:
                print(weight_path)
                self.load_weight(weight_path)
                self.evaluate_classifier()
