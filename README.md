# Pytorch 實作系列

[english-version](https://github.com/gitE0Z9/pytorch-implementations/blob/main/README.en.md)

[中文版本](https://github.com/gitE0Z9/pytorch-implementations/blob/main/README.md)

此專案初衷是實作多年來學習的深度學習模型，並提供 notebook 跟 medium 文章學習，而非一個專注於特定領域的 production-ready 庫

接下來的開發方向會往 monorepo 延伸

## 套件

### 一般

套件名: `common`

| Model      | Article Link                                                                                                  | Package        |
| ---------- | ------------------------------------------------------------------------------------------------------------- | -------------- |
| K-Means    | -                                                                                                             | ``kmeans``     |
| Kernel PCA | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a) | ``kernel_pca`` |

### 圖像分類

套件名: `image_classification`

| Model                                      | Article Link                                                                                                                          | Package                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| ResNet50, ResNet-B, ResNet-C, ResNet-D     | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B8%83-resnet-690868d7af43) | ``resnet``             |
| ResNeXt50-32x4d                            | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-resnext-019a1528cfd7)                     | ``resnext``            |
| ResNeSt50                                  | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-senet-sknet-resnest-273954c83197)        | ``resnest``            |
| Res2Net50                                  | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-res2net-4287e5507a24)                    | ``res2net``            |
| SE-ResNet50, SE-ResNeXt50                  | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-senet-sknet-resnest-273954c83197)        | ``senet``              |
| SKNet                                      | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-senet-sknet-resnest-273954c83197)        | ``sknet``              |
| Residual attention network(AttentionNet56) | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-senet-and-its-variants-2-f8f77cef8e2b)   | ``residual_attention`` |
| DenseNet                                   | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-densenet-467fbf0ce976)                   | ``densenet``           |
| Highway Network                            | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-highway-network-1e8bd63e432f)            | ``highway``            |
| ConvNeXt                                   | -                                                                                                                                     | ``convnext``           |
| MobileNet v1                               | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-mobilenet-v1-v2-9224c02ff45e)             | ``mobilenet``          |
| MobileNet v2                               | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-mobilenet-v1-v2-9224c02ff45e)             | ``mobilenetv2``        |
| MobileNet v3                               | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-mobilenet-v3-e1a90b8a9abc)               | ``mobilenetv3``        |
| Xception                                   | -                                                                                                                                     | ``xception``           |
| EfficientNet v1                            | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-efficientnet-v1-v2-bdd18eb59b3a)          | ``efficientnet``       |
| EfficientNet v2                            | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-efficientnet-v1-v2-bdd18eb59b3a)          | ``efficientnetv2``     |
| GhostNet                                   | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-ghostnet-10b0bab4110e)                   | ``ghostnet``           |
| GhostNet v2                                | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-ghostnetv2-v3-202529dd7671)              | ``ghostnetv2``         |
| GhostNet v3                                | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-ghostnetv2-v3-202529dd7671)              | ``ghostnetv3``         |
| ShuffleNet                                 | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-shufflenet-v1-v2-c37ff4c3197d)            | ``shufflenet``         |
| ShuffleNetV2                               | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-shufflenet-v1-v2-c37ff4c3197d)            | ``shufflenetv2``       |
| SqueezeNet                                 | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-squeezenet-squeezenext-45049b438316)      | ``squeezenet``         |
| SqueezeNeXt                                | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-squeezenet-squeezenext-45049b438316)      | ``squeezenext``        |
| Extraction                                 | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov1-a56b49024c22)                       | ``extraction``         |
| DarkNet19                                  | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov2-722c4d66cd43)                       | ``darknet19``          |
| DarkNet53                                  | -                                                                                                                                     | ``darknet53``          |
| Vision transformer (ViT)                   | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-vit-11ecba1796a3)                         | ``vit``                |
| Distillated Vision transformer (DeiT)      | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-deit-f93bb1d46d21)                        | ``deit``               |

### 物件偵測

套件名: `object_detection`

| Model                               | Article Link                                                                                                    | Package         |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------- |
| You only look once (YOLO)           | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov1-a56b49024c22) | ``yolov1``      |
| Tiny YOLOv1                         | -                                                                                                               | ``yolov1_tiny`` |
| YOLOv2                              | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov2-722c4d66cd43) | ``yolov2``      |
| Tiny YOLOv2                         | -                                                                                                               | ``yolov2_tiny`` |
| YOLOv3                              | -                                                                                                               | ``yolov3``      |
| Tiny YOLOv3                         | -                                                                                                               | ``yolov3_tiny`` |
| Single shot multibox detector (SSD) | -                                                                                                               | ``ssd``         |
| RetinaNet                           | -                                                                                                               | ``retinanet``   |

### 語義分割

套件名: `semantic_segmentation`

| Model                                       | Article Link                                                                                                                        | Package            |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| UNet                                        | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%8D%81-unet-545efa00ad99) | ``unet``           |
| Fully convolution network (FCN)             | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-fcn-89cac059179b)                       | ``fcn``            |
| Pyramid spatial pooling network (PSPNet)    | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-pspnet-8059dc329221)                    | ``pspnet``         |
| Dual attention(DANet)                       | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-dual-attention-56013cbf927a)            | ``dual_attention`` |
| DeepLab v1                                  | -                                                                                                                                   | ``deeplabv1``      |
| DeepLab v2                                  | -                                                                                                                                   | ``deeplabv2``      |
| DeepLab v3                                  | -                                                                                                                                   | ``deeplabv3``      |
| Lite reduced ASPP (LR-ASPP)                 | -                                                                                                                                   | ``lr_aspp``        |
| Reduced ASPP (R-ASPP)                       | -                                                                                                                                   | ``r_aspp``         |
| DenseCRF                                    | -                                                                                                                                   | ``densecrf``       |
| Multi-scale context aggregation by dilation | -                                                                                                                                   | ``mscad``          |
| ParseNet                                    | -                                                                                                                                   | ``parsenet``       |
| Segmentation transformer (SETR)             | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-setr-0091abf13f82)                     | ``setr``           |

### 3D辨識

套件名: `classification_3d` (unstable naming)

| Model    | Article Link | Package      |
| -------- | ------------ | ------------ |
| PointNet | -            | ``pointnet`` |

### 光學文字辨識

套件名: `ocr`

| Model                                       | Article Link                                                                                                    | Package  |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------- |
| Convolution recurrent neural network (CRNN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-crnn-b2a7a8fa1698) | ``crnn`` |

### 姿態偵測

套件名: `pose_estimation`

| Model             | Article Link                                                                                                                        | Package       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Stacked hourglass | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-stacked-hourglass-network-ba7ef1ea0e73) | ``hourglass`` |

### 圖像生成

套件名: `image_generation`

| Model                                | Article Link                                                                                                                                                  | Package            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Variational autoencoder (VAE)        | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%BA%8C-variational-autoencoder-954596aae539)        | ``vae``            |
| Generative adversarial network (GAN) | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B8%89-generative-adversarial-network-445ffdc297fd) | ``gan``            |
| Deep convolution GAN (DCGAN)         | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-11-dcgan-40a78e279030)                                 | ``dcgan``          |
| Conditional GAN (cGAN)               |                                                                                                                                                               | ``cgan.ipynb``     |
| PixelRNN - diagonal bilstm           |                                                                                                                                                               | ``pixelrnn``       |
| PixelRNN - row lstm                  |                                                                                                                                                               | ``pixelrnn``       |
| PixelCNN                             |                                                                                                                                                               | ``pixelcnn``       |
| Gated PixelCNN                       |                                                                                                                                                               | ``gated_pixelcnn`` |

### 風格遷移

套件名: `style_transfer`

| Model                                   | Article Link                                                                                                                                        | Package                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| Neural style transfer                   | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B9%9D-image-style-transfer-371e161c5620) | ``neural_style_transfer`` |
| Neural Doodle                           | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-neural-doodle-80bb55108836)                             | ``neural_doodle``         |
| Fast style transfer                     | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-fast-style-transfer-6630af677395)                      | ``fast_style_transfer``   |
| Adaptive instance normalization (AdaIN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-adain-f18fd4bca76b)                                    | ``adain``                 |
| Pix2Pix                                 | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-14-pix2pix-5b550c1fbb39)                     | ``pix2pix``               |

### 序列資料

套件名: `sequence_data`

| Model                              | Article Link                                                                                                                                           | Package         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| Long and short term memory (LSTM)  | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%9B%9B-long-short-term-memory-21c097616641)  | ``lstm``        |
| Bidirectional LSTM (BiLSTM)        | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-bilstm-92d8e01d488e)                                      | ``lstm``        |
| Gated recurrent unit (GRU)         | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gru-8510d5bf3261)                                          | ``gru``         |
| Temporal convolution network (TCN) | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-12-temporal-convolutional-network-799a243ffa2d) | ``tcn``         |
| LSTNet                             | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-lstnet-4cd561f114a3d)                                     | ``lstnet``      |
| Seq2Seq                            | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%85%AD-sequence-to-sequence-327886dafa4)     | ``seq2seq``     |
| Transformer                        | -                                                                                                                                                      | ``transformer`` |

### 語言模型

套件名: `language_model`

| Model                                           | Article Link                                                                                                    | Package      |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------ |
| Word2vec                                        | [medium](https://acrocanthosaurus627.medium.com/language-model-from-scratch-with-pytorch-word2vec-10e77770cc57) | ``word2vec`` |
| GloVe                                           | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a)   | ``glove``    |
| Vector log-bilinear language model (vLBL/ivLBL) | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a)   | ``vlbl``     |

### 文本分類

套件名: `text_classification`

| Model                   | Article Link                                                                                                                            | Package      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| TextCNN                 | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%BA%94-textcnn-cd9442139f8c)  | ``textcnn``  |
| Character CNN (CharCNN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-charcnn-47020fdc76d4)                      | ``charcnn``  |
| Very deep CNN (VDCNN)   | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-vdcnn-0bfdf5681d45)                         | ``vdcnn``    |
| Recurrent CNN (RCNN)    | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-rcnn-for-text-classification-17880a540591) | ``rcnn``     |
| Dynamic CNN (DCNN)      | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-dcnn-a9241a1ff418)                         | ``dcnn``     |
| FastText                | -                                                                                                                                       | ``fasttext`` |

### 標籤預測

套件名: `tag_prediction`

| Model                                                      | Article Link                                                                                                          | Package        |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------- |
| Bidirectional LSTM - Conditional random field (BiLSTM-CRF) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-bilstm-crf-7d2014a286f6) | ``bilstm_crf`` |

### 文本生成

套件名: `text_generation`

| Model                 | Article Link                                                                                                                   | Package              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| Show and tell         | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-show-and-tell-attend-2638dce945fe) | ``show_and_tell``    |
| Show, attend and tell | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-show-and-tell-attend-2638dce945fe) | ``show_attend_tell`` |

### 少樣本學習

 套件名: `few_shot`

| Model                | Article Link                                                                                                                                   | Package          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Siamese network      | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B8%80-siamese-network-c06dc78242ed) | ``siamese``      |
| Prototypical network | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-13-prototypical-network-360f0e411d21)   | ``prototypical`` |

### 表示學習

套件名: `representation`

| Model                                       | Article Link                                                                                                  | Package       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------- |
| Positive pairwise mutual information (PPMI) | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a) | ``ppmi``      |
| Hellinger PCA                               | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a) | ``hellinger`` |

### 圖神經網路

套件名: `graph`

| Model                               | Article Link                                                                                                   | Package       |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------- |
| Graph Convolution Network (GCN)     | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gcn-c617638a9fcf) | ``gcn``       |
| Graph Attention Network (GAT)       | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gat-a0a413e3cd12) | ``attention`` |
| Graph Attention Network v2 (GAT v2) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gat-a0a413e3cd12) | ``attention`` |

### 增強學習

套件名: `reinforcement_learning`

| Model          | Article Link                                                                                                                                  | Package |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Deep Q Network | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%85%AB-deep-q-network-b12d7769e337) | ``dqn`` |

## 開發環境

`python`: `3.11`

## 安裝方式

```pip install git+https://www.github.com/gite0z9/pytorch-implementations.git@main#torchlake --target=/torchlake```

## 專案結構

`notebooks`: 展示如何使用 torchlake.

`torchlake`: 由不同應用領域組成的深度學習包.

每個領域大致上會有如下結構

``` lang=sh
├───adapter
├───artifacts
│   └───model_name
├───configs
│   └───model_name
├───constants
├───controller
├───datasets
│   └───dataset_name
|       └───datasets.py
├───models
│   ├───base
│   └───model_name
|       |───reference
|       |   └───paper
│       ├───model.py
│       ├───network.py
│       ├───loss.py
│       ├───helper.py
│       └───decode.py
├───runs
├───scripts
│   └───debug
├───tests
├───utils
└───requirements.txt
```

`adapter`: 介接 controller 和其他資源(model, model.loss, etc.)

`configs`: 模型設定檔，包括裝置定義、模型定義、訓練定義、推論定義，共四個面向

`constants`: 固定的值，包括 constant 和 enum

`controller`: 控制器，`controller`是共用基底，`trainer`、`evaluator`、`predictor`負責訓練、評估、預測三種工作

`models`: 模型定義，`network.py`是模型區塊，`model.py`是最後組裝的模型，`loss.py`是損失函數

`datasets`: 資料集，目前是照領域區分，有分 raw dataset 和 csv dataset，前者是讀取 raw data，後者是讀取處理過的 csv(如歸一化座標)

`runs`: 記載`trainer`和`evaluator`的結果的 tensorboard log 資料夾

`tests`: 單元測試，使用 `pytest`

`utils`: 存放依賴性低且復用性高的函式

model 和 dataset 的 config 會用 `pydantic` 控制格式

## 模組依賴關係

![modules drawio](https://github.com/user-attachments/assets/610e83f6-7dc0-4437-a663-29f1ad93bfda)
