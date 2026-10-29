# Quantization Method

## Methods

| Folder/file | Method | Current status |
|---|---|---|
| `cnn/lsq.py` | LSQ | learnable-step quantizer plus Conv2d/Linear wrappers. |
| `cnn/pact.py` | PACT | Learnable activation clipping plus wrappers. |
| `cnn/n2uq.py` | N2UQ | Learnable input thresholds, uniform output levels, and a piecewise G-STE surrogate. |
| `transformer/q_detr.py` | Q-DETR | Query quantization and query distillation loss adapter. |
| `transformer/aq_detr.py` | AQ-DETR | Auxiliary query module and layer-wise feature distillation loss. |
| `transformer/gplq.py` | GPLQ | Activation-first quantizer and two-stage controller. |
| `transformer/vvtq.py` | VVTQ | Running variation statistics and quantization adapter. |
| `transformer/q_mhsa.py` | Q-MHSA | CCM and LSC core module. |

## Smoke test

```bash
python -c "import torch; from quant_methods.cnn import LSQFakeQuantizer; x=torch.randn(2,8,16,requires_grad=True); y=LSQFakeQuantizer(4)(x); y.mean().backward(); print(y.shape, x.grad is not None)"
```

## Detailed hyperparameter settings and tuning procedures

All compared methods use the same FP32 initialization,  data split, optimizer budget, random seeds, and evaluation protocol. Method-specific hyperparameters are selected on the validation set using a small pre-defined grid search. The final test set is not used for hyperparameter selection.

For LSQ-based optimization strategies, the search mainly covers the learning rate multiplier, schedule transition point, EMA decay, Hessian update interval, target transition rate, and QAT fraction. For activation quantizers, the grid covers the clipping/step-size initialization and the corresponding regularization coefficient. For transformer-oriented baselines, the temperature, distillation weight, and controller momentum are tuned in the same manner.

The selected configuration is fixed before the final training run and is then used unchanged for all repeated trials. Detailed candidate values and selected values are stored together with each experiment checkpoint and training log.
