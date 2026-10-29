# Project File Description

This directory contains the training scripts, quantization modules, comparison methods, and implementations of our method.

## Main Files

| File or Directory | Description |
|---|---|
| `train.py` | Training script for model training, quantization-aware training, validation, and checkpoint saving. |
| `yolov5s.yaml` | YOLOv5s model architecture configuration file. |
| `quant_methods/`  | Reimplemented quantization methods and comparison optimization methods. |
| `our method/`     | Implementation of our method, including Q-GBFusion and Q-ADA. |
| `checkpoint`      | Full-precision model weights.                                |

## Method Directory Description

- `LTQ/`: Entry point for replacing quantization operators, including convolutional modules, with quantized modules.
- `quant_methods/`: Reimplemented or adapted comparison quantization methods, providing unified quantizer and training optimization interfaces.
- `our method/`: Our proposed method, including the feature-fusion gradient-balancing module Q-GBFusion and the saliency-alignment module Q-ADA.
