# Classification models

It is based on flax's ImageNet classification sample.
- https://github.com/google/flax/tree/main/examples/imagenet

## Train

### Requirements
- TensorFlow dataset `imagenette/full-size-v2:1.*.*`

### Running locally
```
python main.py --workdir=./train_dir --config=configs/default.py
```

### Models

| Model                       | Args                               |
| :-------------------------- | :--------------------------------- |
|MobileNetV1 alpha=1.0 depth depth multiplier=1.0 | `--config.model=MobileNetV1_10` |
| MobileNetV2 alpha=1.0       | `--config.model=MobileNetV2_10`    |
| MobileNetV3 Large alpha=1.0 | `--config.model=MobileNetV3_Large` |
| MobileNetV3 Small alpha=1.0 | `--config.model=MobileNetV3_Small` |