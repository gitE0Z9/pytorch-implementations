from yolov2.controller import Controller
from tqdm import tqdm

control = Controller(
    cfg_path="config/resnet18.yml", dataset_name="VOC", mode="DETECTOR"
)

control.set_preprocess("train", 416)
control.load_data("train")

for i, (img, label) in enumerate(control.data["train"]["dataset"]):
    if len(label) == 0:
        print(i, len(label))


for i, (img, label) in tqdm(enumerate(control.data["train"]["loader"])):
    if len(label) == 0:
        print(i, len(label))
