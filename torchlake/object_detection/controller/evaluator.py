from pathlib import Path
from typing import Iterator

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from torchlake.object_detection.metrics.map import MeanAveragePrecision

from ..constants.schema import DetectorContext
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
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, int]], pd.DataFrame]:
        """evaluate detector with mAP

        Args:
            predictor (Predictor): predictor class
            model (nn.Module): detector
            data (Iterator): data loader
            class_names (list[str]): class names

        Returns:
            tuple[dict[str, Tuple[List[float], List[float], float]], pd.DataFrame]: precision, recall, AP@0.5
        """
        metric = MeanAveragePrecision(self.context, class_names)

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
            metric.update(detections, labels)

        return metric.ap_table, metric.score()

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
