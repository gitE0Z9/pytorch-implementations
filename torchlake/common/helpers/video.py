from typing import Generator

import cv2
import numpy as np
from tqdm import tqdm


class VideoReader:
    def __init__(self, path: str):
        """Helper for video reading

        Args:
            path (str): video path
        """
        self.handle = cv2.VideoCapture(path)

    def __len__(self) -> int:
        return int(self.handle.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.handle.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.handle.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

    @property
    def fps(self) -> float:
        return self.handle.get(cv2.CAP_PROP_FPS)

    def read(self):
        return self.handle.read()

    def release(self):
        return self.handle.release()


class VideoWriter:
    def __init__(
        self,
        path: str,
        encode_format: str,
        fps: float,
        shape: tuple[int, int],
    ):
        """Helper for video writing

        Args:
            path (str): output path
            encode_format (str): video encoded format, (avi: XVID, mp4: MJPG, H264)
            fps (float): frame per second
            shape (tuple[int, int]): video shape, in format of (width, height)
        """
        fourcc = cv2.VideoWriter_fourcc(*encode_format)
        self.fps = fps
        self.shape = shape
        self.handle = cv2.VideoWriter(path, fourcc, fps, shape)

    def run(self, img_generator: Generator[np.ndarray, None, None]):
        for img in tqdm(img_generator):
            self.handle.write(img)

    def write(self, frame: np.ndarray):
        return self.handle.write(frame)

    def release(self):
        return self.handle.release()
