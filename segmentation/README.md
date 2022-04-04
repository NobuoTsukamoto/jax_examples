# Semantic Segmentation

## Train

### Requirements
- TensorFlow dataset `cityscapes:1.*.*`  
  See [cityscapes](https://www.tensorflow.org/datasets/catalog/cityscapes) in TensorFlow dataset for dataset preparation.

### Running locally
```
python main.py --workdir=./fast_scnn_cityscapes --config=configs/default.py
```

### Models

| Model     | Args                       |
| :-------- | :------------------------- |
| Fast-SCNN | `--config.model=Fast_SCNN` |