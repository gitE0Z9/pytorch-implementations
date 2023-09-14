# Library

A library for object detection.

## Supported models

| model    | status |
| -------- | ------ |
| YOLOv1   | ✔️      |
| YOLOv2   | ✔️      |
| YOLO9000 | ❌      |
| SSD      | ❌      |

## Config

1. Set dataset config. In `datasets/<dataset_name>/config.yml`, change `ROOT` to where the dataset stored.

* Field explanation

    `ROOT`: the path to the folder of the dataset.
    `CSV_ROOT`: the path to the csv file contained labels. Column ordering is `cx`, `cy`, `w`, `h`, `class_id`.
    `CLASSES_PATH`: the path to the file contained names of classes.
    `NUM_CLASSES`: the number of classes in this dataset.

2. Set model config. In `models/<detector_name>/<backbone_name>.yml`, set value you prefered.

* Field explanation

    `HARDWARE.DEVICE`: assign the device to compute.
    `HARDWARE.NUM_WORKERS`: the number of workers of the dataloader.
    `HARDWARE.AMP`: enable automatic mixed precision during training.
    `MODEL.NAME`: the name of the detector.
    `MODEL.BACKBONE`: the backbone of the detector.
    `MODEL.NUM_ANCHORS`: the number of anchors of each output layer, specified in list format for multiscale prediction.
    `MODEL.SCALE`: the downsample scale of each output layer, specified in list format for multiscale prediction.
    `MODEL.CLASSIFIER_PATH`: the path to the weight file of the classifier, used for finetuning on the checkpoint of the classifier or loaded as the pretrained weight of the detector.
    `MODEL.DETECTOR_PATH`: the path to the weight file of the detector, used for finetuning on the checkpoint of the detector.
    `TRAIN.DETECTOR.IMAGE_SIZE`: the input size of image.
    `TRAIN.DETECTOR.BATCH_SIZE`: the batch size of data.
    `TRAIN.DETECTOR.ACC_ITER`: the times of gradient accumulation, multiply with `TRAIN.DETECTOR.BATCH_SIZE` for desired batch size.
    `TRAIN.DETECTOR.OPTIM.TYPE`: the name of optimizer, supports `adam`, `sgd`.
    `TRAIN.DETECTOR.OPTIM.LR`: the learning rate of the optimizer.
    `TRAIN.DETECTOR.OPTIM.DECAY`: the weight decay of the optimizer, worked as the regularization parameter.  
    `TRAIN.DETECTOR.OPTIM.MOMENTUM`: the momentum of the optimizer `sgd`.  
    `TRAIN.DETECTOR.START_EPOCH`: the epoch to start training.
    `TRAIN.DETECTOR.END_EPOCH`: the epoch to end training.
    `TRAIN.DETECTOR.MULTISCALE`: enable multiscale training for YOLOv2.
    `TRAIN.DETECTOR.SAVE.DIR`: the path to the folder of the detector weight file.
    `TRAIN.DETECTOR.SAVE.INTERVAL`: the interval of epoch to save the detector weight file.
    `TRAIN.CLASSIFIER.IMAGE_SIZE`: the input size of image.
    `TRAIN.CLASSIFIER.BATCH_SIZE`: the batch size of data. 
    `TRAIN.CLASSIFIER.ACC_ITER`: the times of gradient accumulation, multiply with `TRAIN.CLASSIFIER.BATCH_SIZE` for desired batch size.
    `TRAIN.CLASSIFIER.OPTIM.TYPE`: the name of the optimizer, supports `adam`, `sgd`.
    `TRAIN.CLASSIFIER.OPTIM.LR`: the learning rate of the optimizer.
    `TRAIN.CLASSIFIER.OPTIM.DECAY`: the weight decay of the optimizer, worked as the regularization parameter.
    `TRAIN.CLASSIFIER.OPTIM.MOMENTUM`: the momentum of the optimizer `sgd`. 
    `TRAIN.CLASSIFIER.START_EPOCH`: the epoch to start training.
    `TRAIN.CLASSIFIER.END_EPOCH`: the epoch to end training.
    `TRAIN.CLASSIFIER.FINETUNE.EPOCH`: the number of epoches for finetuning.
    `TRAIN.CLASSIFIER.FINETUNE.IMAGE_SIZE`: the input size of image for finetuning.
    `TRAIN.CLASSIFIER.FINETUNE.BATCH_SIZE`: the batch size of data for finetuning.
    `TRAIN.CLASSIFIER.SAVE.DIR`: the path to the folder of the classifier weight file.
    `TRAIN.CLASSIFIER.SAVE.INTERVAL`: the interval of epoch to save the classifier weight file.
    `INFERENCE.METHOD`: the name of the preprocessing method, default value is `torchvision`, other supported method includes `greddy`, `soft`, `fast`, `diou`, `confluence`. If left the value empty, only filter predictions by the probability.
    `INFERENCE.CONF_THRESH`: the confidence threshold for NMS, the higher, the more confident predictions.
    `INFERENCE.NMS_THRESH`: the iou threshold for NMS, the lower, the more separated predictions.
    `INFERENCE.PARAMETER.CONFLUENCE_THRESH`: optional, the confluence threshold for confluence NMS, the higher, the more separated predictions.
    `INFERENCE.PARAMETER.SIGMA`: optional, the sigma parameter for soft NMS, the higher, the more separated predictions.
    `INFERENCE.PARAMETER.BETA`: optional, the beta parameter for diou NMS, the higher, the method will degenerated back to greedy NMS.

## Usage

### Training

`python train.py -c configs/yolov2/resnet18.yml -nt DETECTOR -d VOC --stage scratch -desc <desc of this run>`

### Finetuning

`python train.py -c configs/yolov2/resnet18.yml -nt DETECTOR -d VOC --stage finetune -desc <desc of this run>`

### Evaluation

`python eval.py -c configs/yolov2/resnet18.yml -nt DETECTOR -d VOC -w artifacts/yolov2/resnet18/yolov2.resnet18.60.pth -s -desc <desc of this run>`

### Prediction

`python predict.py -c configs/yolov2/resnet18.yml -nt DETECTOR -d VOC -f "C://Users/user/Desktop/demo.jpg" -media image -w artifacts/yolov2/resnet18/yolov2.resnet18.60.pth -show`

## Parameters

### Model

* `config`: path to model config yaml file.

* `dataset`: dataset name for class.

### Lifecycle

* `mode`: running mode, `train` or `test`.

* `stage`: model lifecycle stage, `scratch`, `finetune`, or `inference`.

* `network_type`: `detector` or `classifier`.

### Inference

* `weight`: path to model weight.

* `media_type`: media type, `image` or `video`.

* `files`: path to files.

* `show`: show results or not, default `false`.

* `save_dir`: directory to save results.

## TODO

[] cfg inherit

[] segmentation
