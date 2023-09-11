# Scripts

First please move certain script into project root directory.

- `prepare_data_voc.py`

  - `cmd`: `python prepare_data_voc.py`

  - `purpose`: generate VOC csv file from raw data 


- `prepare_data_coco.py`

  - `cmd`: `python prepare_data_coco.py`

  - `purpose`: generate COCO csv file from raw data 


- `yolo_anchors.py`

  - `cmd`: `python yolo_anchors.py -c configs/yolov2/resnet18.yml -d VOC --show --save-dir <directory to save anchors.txt>`

  - `purpose`: generate yolov2 anchors with k-means
