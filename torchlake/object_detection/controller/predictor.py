from pathlib import Path
from typing import List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchlake.common.utils.image import load_image

from ..constants.enums import NetworkType, OperationMode
from ..controller.controller import Controller
from ..utils.inference import model_predict
from ..utils.plot import draw_pred


class Predictor(Controller):
    def set_preprocess(self, input_size: int):
        """
        Set preprocessing pipeline.
        Be careful, some transformations might drop targets.
        """
        preprocess = A.Compose(
            [
                A.Resize(input_size, input_size),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ],
        )

        self.data[OperationMode.TEST.value]["preprocess"] = preprocess

    def detect_single_image(self, image: np.ndarray, transform) -> list[torch.Tensor]:
        img_h, img_w, _ = image.shape

        image = transform(image=image)["image"].to(self.device)
        output = model_predict(self.model, image)

        if self.network_type == NetworkType.DETECTOR.value:
            return self.postprocess(output, (img_h, img_w))[0]
        else:
            return output

    def predict_image_file(
        self,
        weight_path: str,
        image_paths: List[str],
        show: bool = False,
        save_dir: str = None,
    ):
        transform = self.prepare_inference()
        self.load_weight(weight_path)

        assert isinstance(image_paths, list), "image should be a list."

        for image_path in image_paths:
            original_image = load_image(image_path, is_numpy=True)
            detections = self.detect_single_image(original_image, transform)

            if self.network_type == NetworkType.DETECTOR.value:
                copied_image = original_image.copy()
                draw_pred(
                    copied_image,
                    detections,
                    class_names=self.class_names,
                    class_show=True,
                    class_colors=self.palette,
                )

                print(image_path, len(detections))

                if show:
                    plt.imshow(copied_image)
                    plt.show()

                if save_dir:
                    filename = Path(image_path).name
                    dst = Path(save_dir).joinpath(filename)
                    cv2.imwrite(dst.as_posix(), copied_image[:, :, ::-1])

            elif self.network_type == NetworkType.CLASSIFIER.value:
                cls_idx = detections.argmax(1).item()
                print(image_path, self.class_names[cls_idx])

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
            filename = Path(video_path).name
            dst = Path(save_dir).joinpath(filename)
            ext = Path(video_path).suffix.replace(".", "")
            if ext == "avi":
                encoder = cv2.VideoWriter_fourcc(*"XVID")
            elif ext == "mp4":
                encoder = cv2.VideoWriter_fourcc(*"H264")
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
                class_names=self.class_names,
                class_show=True,
                class_colors=self.palette,
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
