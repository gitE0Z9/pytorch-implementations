import os
from typing import List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from constants.enums import NetworkType, OperationMode
from utils.inference import model_predict
from utils.plot import draw_pred, load_image

from controller.controller import Controller


class Predictor(Controller):
    def set_preprocess(self, input_size: int):
        """
        Set preprocessing pipeline.
        Be careful, some transformations might drop targets.
        """
        preprocess = A.Compose(
            [
                A.Resize(input_size, input_size),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ],
        )

        self.data[OperationMode.TEST.value]["preprocess"] = preprocess

    def detect_single_image(self, image: np.ndarray, transform) -> torch.Tensor:
        img_h, img_w, _ = image.shape

        image = transform(image=image)["image"].to(self.device)
        output = model_predict(self.model, image)
        return self.postprocess(output, (img_h, img_w))[0]

    def predict_image_file(
        self,
        model_path: str,
        image_paths: List[str],
        show: bool = False,
        save_dir: str = None,
    ):
        transform = self.prepare_inference()
        self.load_weight(model_path)

        assert isinstance(image_paths, list), "image should be a list."

        for image_path in image_paths:
            original_image = load_image(image_path)
            detections = self.detect_single_image(original_image, transform)

            if self.network_type == NetworkType.DETECTOR.value:
                copied_image = original_image.copy()
                draw_pred(
                    copied_image,
                    detections,
                    class_name=self.class_names,
                    class_show=True,
                    class_color=self.palette,
                )

                print(image_path, len(detections))

                if show:
                    plt.imshow(copied_image)
                    plt.show()

                if save_dir:
                    filename = os.path.basename(image_path)
                    dst = os.path.join(save_dir, filename)
                    cv2.imwrite(dst, copied_image)

            elif self.network_type == NetworkType.CLASSIFIER.value:
                output = output.argmax(1)
                print(image_path, self.class_names[output.item()])

    def predict_video_file(
        self,
        model_path: str,
        video_path: str,
        show: bool = False,
        save_dir: str | None = None,
    ):
        """Press Q to quit"""
        transform = self.prepare_inference()
        self.load_weight(model_path)

        vid = cv2.VideoCapture(video_path)

        # writing video
        if save_dir:
            dst = os.path.join(save_dir, os.path.basename(video_path))
            ext = os.path.basename(video_path).split(".")[-1]
            if ext == "avi":
                encoder = cv2.VideoWriter_fourcc(*"XVID")
            elif ext == "mp4":
                encoder = cv2.VideoWriter_fourcc(*"MP4V")
            else:
                raise NotImplementedError
            writer = cv2.VideoWriter(
                dst, encoder, vid.get(5), (int(vid.get(3)), int(vid.get(4)))
            )

        # show window
        if show:
            cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

        while 1:
            returned, original_image = vid.read()
            if not returned:
                print("Video end")
                break

            detections = self.detect_single_image(original_image, transform)
            copied_image = original_image.copy()
            draw_pred(
                copied_image,
                detections,
                class_name=self.class_names,
                class_show=True,
                class_color=self.palette,
            )

            if show:
                cv2.imshow("video", copied_image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    vid.release()
                    cv2.destroyAllWindows()
                    break

            if save_dir:
                writer.write(copied_image)

        vid.release()
        cv2.destroyAllWindows()

        if save_dir:
            writer.release()
