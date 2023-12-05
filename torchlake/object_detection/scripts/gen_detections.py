import os
from tqdm import tqdm
import pandas as pd

from object_detection.controller.predictor import Predictor
from object_detection.utils.inference import model_predict, yolo_postprocess

if __name__ == "__main__":
    control = Predictor("configs/yolov2/resnet18.yml", "VOC", "DETECTOR", "inference")

    control.load_detector()
    control.set_preprocess("test", control.cfg["TRAIN"][control.mode]["IMAGE_SIZE"])
    control.load_data("test")
    control.model.eval()
    control.load_weight("model/yolov2.resnet18.60.pth")

    data = control.data["test"]["dataset"]

    for index, (img, _) in tqdm(enumerate(data)):
        record = data.table.loc[index]
        if isinstance(record, pd.Series):
            filename = record["name"]
        else:
            filename = record.iloc[0]["name"]
        h, w, _ = data.get_img(filename).shape
        # dst = os.path.basename(filename)
        # img_path = filename.replace("Annotations", "JPEGImages").replace("xml", "jpg")

        # copyfile(filename, f"./data/label/{dst}")
        # copyfile(
        #     img_path,
        #     f"./data/img/{dst.replace('xml', 'jpg')}",
        # )

        img = img.to(control.cfg["HARDWARE"]["DEVICE"])
        output = model_predict(control.model, img)

        detection_result = yolo_postprocess(
            output,
            h,
            w,
            control.cfg["DATA"][control.dataset_name]["NUM_CLASSES"],
            control.anchors.to("cpu"),
            control.cfg["INFERENCE"],
        )[0]

        filename = os.path.basename(filename).split(".")[0]

        with open(f"eval/{filename}.txt", "w") as f:
            for result in detection_result:  # N, 25
                x, y, w, h = result[:4].int().clip(0)
                p, c_index = result[5:].max(0)
                name = control.classes_name[c_index]
                print(f"{name} {p} {x} {y} {w} {h}", file=f)
