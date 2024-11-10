# PyTorch Implementations

[english-version](https://github.com/gitE0Z9/pytorch-implementations/blob/main/README.en.md)

[中文版本](https://github.com/gitE0Z9/pytorch-implementations/blob/main/README.md)

The articles are simultaneously published on my personal blog (https://gite0z9.github.io).

The original intention of this project is to implement deep learning models I have learned over the years, providing notebooks and Medium articles for learning. This project is not a production-ready library for a specific domain but serves as a resource for learning.

The development direction will extend towards a monorepo.

## Packages

### Common

package name: `common`

| Model      | Article Link                                                                                                  | Package        |
| ---------- | ------------------------------------------------------------------------------------------------------------- | -------------- |
| K-Means    | -                                                                                                             | ``kmeans``     |
| Kernel PCA | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a) | ``kernel_pca`` |


### Image classification

package name: `image_classification`

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
| MobileNet v1                               | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-mobilenet-v1-v2-9224c02ff45e)             | ``mobilenet``          |
| MobileNet v2                               | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-mobilenet-v1-v2-9224c02ff45e)             | ``mobilenetv2``        |
| MobileNet v3                               | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-mobilenet-v3-e1a90b8a9abc)               | ``mobilenetv3``        |
| GhostNet                                   | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-ghostnet-10b0bab4110e)                   | ``ghostnet``           |
| GhostNet v2                                | -                                                                                                                                     | ``ghostnetv2``         |
| ShuffleNet                                 | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-shufflenet-v1-v2-c37ff4c3197d)            | ``shufflenet``         |
| ShuffleNetV2                               | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-shufflenet-v1-v2-c37ff4c3197d)            | ``shufflenetv2``       |
| SqueezeNet                                 | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-squeezenet-squeezenext-45049b438316)      | ``squeezenet``         |
| SqueezeNeXt                                | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-squeezenet-squeezenext-45049b438316)      | ``squeezenext``        |
| Extraction                                 | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov1-a56b49024c22)                       | ``extraction``         |


### Object detection

package name: `object_detection`

| Model                     | Article Link                                                                                                    | Package    |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------- |
| You only look once (YOLO) | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov1-a56b49024c22) | ``yolov1`` |
| YOLOv2                    | [medium](https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov2-722c4d66cd43) | ``yolov2`` |



### Semantic Segmentation

package name: `semantic_segmentation`

| Model                                    | Article Link                                                                                                                        | Package            |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| UNet                                     | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%8D%81-unet-545efa00ad99) | ``unet``           |
| Fully convolution network (FCN)          | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-fcn-89cac059179b)                       | ``fcn``            |
| Pyramid spatial pooling network (PSPNet) | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-pspnet-8059dc329221)                    | ``pspnet``         |
| Dual attention(DANet)                    | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-dual-attention-56013cbf927a)            | ``dual_attention`` |
| DeepLab v1                               | -                                                                                                                                   | ``deeplabv1``      |
| DeepLab v2                               | -                                                                                                                                   | ``deeplabv2``      |
| DeepLab v3                               | -                                                                                                                                   | ``deeplabv3``      |



### Optical character recognition (OCR)

package name: `ocr`

| Model                                       | Article Link                                                                                                    | Package  |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------- |
| Convolution recurrent neural network (CRNN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-crnn-b2a7a8fa1698) | ``crnn`` |


### Image generation

package name: `image_generation`

| Model                                | Article Link                                                                                                                                                  | Package   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Variational autoencoder (VAE)        | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%BA%8C-variational-autoencoder-954596aae539)        | ``vae``   |
| Generative adversarial network (GAN) | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B8%89-generative-adversarial-network-445ffdc297fd) | ``gan``   |
| Deep convolution GAN (DCGAN)         | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-11-dcgan-40a78e279030)                                 | ``dcgan`` |


### Style transfer

package name: `style_transfer`

| Model                                   | Article Link                                                                                                                                        | Package                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| Neural style transfer                   | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B9%9D-image-style-transfer-371e161c5620) | ``neural_style_transfer`` |
| Neural Doodle                           | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-neural-doodle-80bb55108836)                             | ``neural_doodle``         |
| Fast style transfer                     | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-fast-style-transfer-6630af677395)                      | ``fast_style_transfer``   |
| Adaptive instance normalization (AdaIN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-adain-f18fd4bca76b)                                    | ``adain``                 |
| Pix2Pix                                 | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-14-pix2pix-5b550c1fbb39)                     | ``pix2pix``               |


### Sequence data

package name: `sequence_data`

| Model                              | Article Link                                                                                                                                           | Package     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| Long and short term memory (LSTM)  | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%9B%9B-long-short-term-memory-21c097616641)  | ``lstm``    |
| Bidirectional LSTM (BiLSTM)        | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-bilstm-92d8e01d488e)                                      | ``lstm``    |
| Gated recurrent unit (GRU)         | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gru-8510d5bf3261)                                          | ``gru``     |
| Temporal convolution network (TCN) | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-12-temporal-convolutional-network-799a243ffa2d) | ``tcn``     |
| Seq2Seq                            | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%85%AD-sequence-to-sequence-327886dafa4)     | ``seq2seq`` |


### Language model

package name: `language_model`

| Model                                           | Article Link                                                                                                    | Package      |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------ |
| Word2vec                                        | [medium](https://acrocanthosaurus627.medium.com/language-model-from-scratch-with-pytorch-word2vec-10e77770cc57) | ``word2vec`` |
| GloVe                                           | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a)   | ``glove``    |
| Vector log-bilinear language model (vLBL/ivLBL) | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a)   | ``vlbl``     |


### Text classification

package name: `text_classification`

| Model                   | Article Link                                                                                                                            | Package      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| TextCNN                 | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%BA%94-textcnn-cd9442139f8c)  | ``textcnn``  |
| Character CNN (CharCNN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-charcnn-47020fdc76d4)                      | ``charcnn``  |
| Very deep CNN (VDCNN)   | [medium](https://acrocanthosaurus627.medium.com/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-vdcnn-0bfdf5681d45)                         | ``vdcnn``    |
| Recurrent CNN (RCNN)    | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-rcnn-for-text-classification-17880a540591) | ``rcnn``     |
| Dynamic CNN (DCNN)      | [medium](https://medium.com/@acrocanthosaurus627/pytorch%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-dcnn-a9241a1ff418)                         | ``dcnn``     |
| FastText                | -                                                                                                                                       | ``fasttext`` |

### Tag prediction

package name: `tag_prediction`

| Model                                                      | Article Link                                                                                                          | Package        |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------- |
| Bidirectional LSTM - Conditional random field (BiLSTM-CRF) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-bilstm-crf-7d2014a286f6) | ``bilstm_crf`` |

### Text generation


package name: `text_generation`

| Model         | Article Link | Package           |
| ------------- | ------------ | ----------------- |
| Show and tell | -            | ``show_and_tell`` |


### Few-shot learning

 package name: `few_shot`
| Model                 | Article Link                                                                                                                                   | Package          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Siamese network:      | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E4%B8%80-siamese-network-c06dc78242ed) | ``siamese``      |
| Prototypical network: | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-13-prototypical-network-360f0e411d21)   | ``prototypical`` |


### Representation learning

package name: `representation`

| Model                                       | Article Link                                                                                                  | Package       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------- |
| Positive pairwise mutual information (PPMI) | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a) | ``ppmi``      |
| Hellinger PCA                               | [medium](https://medium.com/@acrocanthosaurus627/language-model-from-scratch-with-pytorch-glove-6dea3f65bc7a) | ``hellinger`` |

            
### Graph neural network (GNN)

package name: `graph`

| Model                           | Article Link                                                                                                   | Package       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------- |
| Graph Convolution Network (GCN) | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gcn-c617638a9fcf) | ``gcn``       |
| Graph Attention Network (GAT)   | [medium](https://acrocanthosaurus627.medium.com/pytorch-%E5%AF%A6%E4%BD%9C%E7%B3%BB%E5%88%97-gat-a0a413e3cd12) | ``attention`` |

### Reinforcement Learning

package name: `reinforcement_learning`

| Model          | Article Link                                                                                                                                  | Package |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Deep Q Network | [medium](https://acrocanthosaurus627.medium.com/%E7%B6%93%E5%85%B8%E7%B6%B2%E8%B7%AF%E7%B3%BB%E5%88%97-%E5%85%AB-deep-q-network-b12d7769e337) | ``dqn`` |


## Installation

```pip install git+https://www.github.com/gite0z9/pytorch-implementations.git@main#torchlake --target=/torchlake```

## Project structure

`notebooks`: demonstrate how to use torchlake.

`torchlake`: deep learning models composed of different domains.

In general, each domain will have a structure similar to the following:

```
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

`adapter`: Interface between the controller and other resources (model, model.loss, etc.).

`configs`: Model configuration files, including device definition, model definition, training definition, and inference definition, covering four aspects.

`constants`: Fixed values, including constants and enums.

`controller`: Controller, where controller is a shared base, and trainer, evaluator, predictor are responsible for training, evaluation, and prediction tasks, respectively.

`models`: Model definitions, where network.py represents the model blocks, model.py is the assembled model, and loss.py is the loss function.

`datasets`: Datasets, currently categorized by domain, with raw datasets and CSV datasets. The former reads raw data, while the latter reads processed CSV data (such as normalized coordinates).

`runs`: Folder documenting TensorBoard logs for trainer and evaluator results.

`tests`: Unit tests using pytest.

`utils`: Stores functions with low dependency and high reusability.

Model and dataset configurations are controlled using pydantic.

## Module dependency

![modules drawio](https://github.com/user-attachments/assets/610e83f6-7dc0-4437-a663-29f1ad93bfda)
