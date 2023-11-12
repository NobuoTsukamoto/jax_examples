# Jax Examples
Jax, Flax, examples (ImageClassification, SemanticSegmentation, and more...)

# About

I started to learn JAX, Flax, Optax, etc ...  
I will be adding mainly computer vision tasks. I will start with code for model learning, inference, and export to other frameworks (such as TensorFlow).

## TODO

- [ ] Implementation of inference code.
- [ ] Export TensorFlow Saved Mdoel or ONNX model, etc...
- [ ] Add more models...
  - [x] Segmentation model (LR-RASPP).
  - [ ] Object detection model.
  - [ ] GAN model.
- [ ] Training with Colab TPU.


# Reference
- [JAX: Autograd and XLA](https://github.com/google/jax)
- [Flax: A neural network library and ecosystem for JAX designed for flexibility](https://github.com/google/flax)
- [Optax](https://github.com/deepmind/optax)

# Models

- [Classification](classification)
- [Semantic Segmentation](segmentation)

## Classification Task

| Paper's | URL |
|:-- |:--|
| MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications | https://arxiv.org/abs/1704.04861 |
| MobileNetV2: Inverted Residuals and Linear Bottlenecks | https://arxiv.org/abs/1801.04381 |
| Searching for MobileNetV3 | https://arxiv.org/abs/1905.02244 |
| Identity Mappings in Deep Residual Networks | https://arxiv.org/abs/1603.05027 |

## Semantic Segmentation Task

| Paper's | URL |
|:-- |:--|
| Fast-SCNN: Fast Semantic Segmentation Network | https://arxiv.org/abs/1902.04502 |
| Searching for MobileNetV3 | https://arxiv.org/abs/1905.02244 |
| Fully Convolutional Networks for Semantic Segmentation | https://arxiv.org/abs/1411.4038 |
| Simple and Efficient Architectures for Semantic Segmentation | https://arxiv.org/abs/2206.08236 |
| DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation | https://arxiv.org/abs/1907.11357 |
| Training Region-based Object Detectors with Online Hard Example Mining | https://arxiv.org/abs/1604.03540 |
| Striking the Right Balance: Recall Loss for Semantic Segmentation | https://arxiv.org/abs/2106.14917 |

# Installation

W.I.P
```
$ pip install jax flax ml_collections clu
```
