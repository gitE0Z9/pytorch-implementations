# Library

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

## Lifecycle

* `mode`: running mode, `train` or `test`.

* `stage`: model lifecycle stage, `scratch`, `finetune`, or `inference`.

* `network_type`: `detector` or `classifier`.

## Inference

* `weight`: path to model weight.

* `media_type`: media type, `image` or `video`.

* `files`: path to files.

* `show`: show results or not, default `false`.

* `save_dir`: directory to save results.

## TODO

[] cfg inherit

[] segmentation
