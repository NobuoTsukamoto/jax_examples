# Classification models

It is based on flax's ImageNet classification sample.
- https://github.com/google/flax/tree/main/examples/imagenet

## Requirements
- GPU backend
    - [Dockerfile](../docker/Dockerfile)
- TPU backend
    - [Google Colab notebook](./notebook/train_image_classification_model_tpu.ipynb)
    - Install dependency  
      ```
      pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      pip install flax
      pip install ml_collections clu
      pip install tensorflow tensorflow_datasets tensorboard
      pip install tf-models-official
      ```

## Preparation

```
git clone --depth 1 https://github.com/NobuoTsukamoto/jax_examples.git
cd jax_example
export PYTHONPATH=`pwd`/common:$PYTHONPATH
cd classification/
```

## Train

### Running locally
```
python main.py \
    --task train \
    --config configs/_CONFIG_FILE_.py \
    --workdir `full path for workdir`
```

### Models

imagenet2012

| Model | Backend | Config | Top-1 accuracy | Epochs | Total params | Note |
| :--   | :--     | :--  | --: | --: | --: | :-- |
| MobileNet v2 | TPU v2-8 | [config](./configs/imagenet_mobilenet_v2_tpu.py) | 71.84% | 499 | 3,538,984 | [blog post](https://zenn.dev/nbo/scraps/fccbce1806c1c2) |
| MobileNet v3 Small | TPU v2-8 | [config](./configs/imagenet_mobilenet_v3_small_tpu.py) | | 1000 | 2,554,968 | [blog post](https://zenn.dev/nbo/scraps/408a03fccbd5f8) |
| MobileNet v3 Large | TPU v2-8 | [config](./configs/imagenet_mobilenet_v3_large_tpu.py) | 74.09% | 698 | 5,507,432 | [blog post](https://zenn.dev/nbo/scraps/95fba843c67189) |
| ResNet50 | TPU v2-8 | [config](./configs/imagenet_resnet50_v1_tpu.py) | 76.3% | 100 | 25,610,152 | |
| ResNet50 Training techniques<br>(ConvNeXt training techniques) | TPU v2-8 | [config](./configs/imagenet_resnet50_v1_training_techniques_tpu.py) | 77.96% | 300 | 25,610,152 | override config<br>--config.batch_size=1024 \ <br> --config.gradient_accumulation_steps=4



## Model summary
```
python main.py --task summarize --config configs/`_MODEL_`.py
```
