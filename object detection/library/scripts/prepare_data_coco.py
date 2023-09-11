import pandas as pd
from datasets.coco.datasets import COCODatasetRaw
from utils.config import load_classes, load_config

cfg = load_config("configs/yolov2/resnet18.yml")
root = cfg["DATA"]["COCO"]["ROOT"]
class_names = load_classes(cfg["DATA"]["COCO"]["CLASSES_PATH"])

dataset = COCODatasetRaw(root, class_names, mode="train")

data = []
for idx, image in enumerate(dataset.labels["images"]):
    anns = dataset.get_label(idx, image["height"], image["width"])
    for ann in anns:
        ann.insert(0, image["file_name"])
        ann.insert(0, idx)
        data.extend(anns)

pd.DataFrame(data).to_csv("coco_train.csv", index=False)
