from pathlib import Path

from ...utils.config import load_classes

WORK_DIR = Path(__file__).parent

IMAGENET_CLASS_NAMES = load_classes(WORK_DIR.joinpath("imagenet.shortnames.txt"))

IMAGENET_DARKNET_CLASS_NAMES = load_classes(
    WORK_DIR.joinpath("imagenet.darknet.shortnames.txt")
)

IMAGENET_CLASS_NOS = load_classes(WORK_DIR.joinpath("imagenet_classes.txt"))

IMAGENET_DARKNET_CLASS_NOS = load_classes(WORK_DIR.joinpath("imagenet_darknet.txt"))
