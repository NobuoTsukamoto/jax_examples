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

| Model | Backend | Config | Top-1 accuracy | Epochs | Note |
| :--   | :--     | :--  | --: | --: | :-- |
| MobileNet v2 | TPU v2-8 | [config](./configs/imagenet_mobilenet_v2_tpu.py) | 71.76 % | 500 ||
| ResNet50 | TPU v2-8 | [config](./configs/imagenet_resnet50_v1_tpu.py) | 76.3 % | 100 |
| ResNet50 Training techniques<br>(ConvNeXt training techniques) | TPU v2-8 | [config](./configs/imagenet_resnet50_v1_training_techniques_tpu.py) | 77.96 % | 300 | override config<br>--config.batch_size=1024 \ <br> --config.gradient_accumulation_steps=4



## Model summary
```
python main.py --task summarize --config configs/`_MODEL_`.py
```
