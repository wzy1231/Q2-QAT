# LSQ-based QAT optimization baselines

| File | Class | Purpose |
|---|---|---|
| `hmqat.py` | `HMQATController` | Adjusts LSQ latent-weight and scale gradients using Hessian/parameter-size sensitivity. |
| `transition_scheduling.py` | `TransitionRateScheduler` | Adjusts the optimization step size according to the actual transition rate of quantized weight states. |
| `qt_dog.py` | `QTDoGSchedule` | FP32 → activation-only → full W4A4 stage scheduling. |
| `ema_qc.py` | `EMALSQController`, `QuantizationCorrection` | EMA smoothing of weights/scales and affine QC on a calibration set. |
| `compute_optimal_qat.py` | `ComputeOptimalQATSchedule` | FP/QAT allocation and cooldown-QAT fusion under a fixed compute budget. |



## Correspondence to the Original Papers and Controlled Adaptations

1. **HMQAT** The original paper includes Hessian/parameter-size sensitivity, mixed-precision search, and quantization-transition-aware scale fine-tuning. Under the fixed W4A4 comparison protocol used here, we retain Hessian-guided updates and scale/latent-parameter sensitivity while disabling mixed-precision search.
2. **TR** Uses the actual transition rate of quantized weight states as feedback and adjusts the next LSQ optimization step according to the target transition rate.
3. **QT-DoG** The original paper studies quantization as a flatness/domain-generalization regularizer and provides an independent implementation. The detection/segmentation reproduction protocol used here additionally specifies 10 epochs of FP32 training, 10 epochs of activation-only quantization, and W4A4 quantization thereafter.
4. **EMA/QC** Implements EMA smoothing and affine correction on a one-epoch calibration set.
5. **Compute-Optimal QAT** Retains fixed-compute stage switching and cooldown-QAT fusion. `qat_fraction` should be pre-registered before the experiment and must not be selected based on test-set results.

## Code Scope

These files provide reusable training controllers.
