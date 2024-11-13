from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from torchlake.common.utils.image import load_image

from ..configs.schema import InferenceCfg
from ..constants.schema import DetectorContext
from ..utils.plot import draw_pred


class Predictor:
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

    def set_postprocess_cfg(self, decoder, inferenceCfg: InferenceCfg = None):
        self.decoder = decoder
        self.inferenceCfg = inferenceCfg

    def postprocess(
        self,
        output: torch.Tensor,
        img_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        return self.decoder.decode(output, img_size)

    def detect_image(
        self,
        model: nn.Module,
        img: np.ndarray,
        transform=None,
        is_batch: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        img_h, img_w = img.shape[1 if is_batch else 0], img.shape[2 if is_batch else 1]

        if transform is not None:
            img = transform(image=img)["image"]

        if not is_batch:
            img = img.unsqueeze(0)

        img = img.to(self.context.device)

        model.eval()
        with torch.no_grad():
            y = model(img).detach().cpu()

        y = self.postprocess(y, (img_h, img_w))

        if not is_batch:
            return y[0]

        return y

    def predict_image_file(
        self,
        model: nn.Module,
        image_paths: list[str],
        class_names: list[str],
        class_colors: dict[str, list[int]] = {},
        transform=None,
        show: bool = False,
        save_dir: str = None,
    ):
        assert isinstance(image_paths, list), "image should be a list."

        for image_path in image_paths:
            original_image = load_image(image_path, is_numpy=True)

            detections = self.detect_image(model, original_image, transform)
            copied_image = original_image.copy()
            draw_pred(
                copied_image,
                detections,
                class_names=class_names,
                class_show=True,
                class_colors=class_colors,
            )

            print(image_path, len(detections))

            if show:
                plt.imshow(copied_image)
                plt.show()

            if save_dir:
                filename = Path(image_path).name
                dst = Path(save_dir).joinpath(filename)
                cv2.imwrite(dst.as_posix(), copied_image[:, :, ::-1])

    def predict_video_file(
        self,
        model: nn.Module,
        video_path: str,
        class_names: list[str],
        class_colors: dict[str, list[int]] = {},
        transform=None,
        show: bool = False,
        save_dir: str | None = None,
    ):
        """Press Q to quit"""
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
                dst.as_posix(),
                encoder,
                vid.get(cv2.CAP_PROP_FPS),
                (
                    int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )

        # show window
        if show:
            cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

        while 1:
            returned, original_image = vid.read()
            if not returned:
                print("Video end")
                break

            detections = self.detect_image(model, original_image, transform)
            copied_image = original_image.copy()
            draw_pred(
                copied_image,
                detections,
                class_names=class_names,
                class_show=True,
                class_colors=class_colors,
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
