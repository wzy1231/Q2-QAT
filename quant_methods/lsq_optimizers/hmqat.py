# -*- coding: utf-8 -*-
"""HMQAT: Hessian-guided LSQ optimization for fixed W4A4 YOLOv5 QAT."""
from __future__ import annotations
from typing import Dict, Iterable
import torch
import torch.nn as nn


class HessianSensitivityEstimator:
    """Hutchinson diagonal-Hessian-trace proxy with a squared-gradient fallback."""
    def __init__(self, damping: float = 1e-8, ema: float = 0.9):
        self.damping, self.ema = float(damping), float(ema)
        self.running: Dict[str, torch.Tensor] = {}

    def estimate(self, loss: torch.Tensor, named_parameters: Iterable):
        params = [(n, p) for n, p in named_parameters if p.requires_grad]
        grads = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True,
                                    create_graph=True, allow_unused=True)
        result = {}
        for (name, p), grad in zip(params, grads):
            if grad is None:
                continue
            try:
                probe = torch.randint_like(grad, 0, 2).mul_(2).sub_(1)
                hv = torch.autograd.grad((grad * probe).sum(), p, retain_graph=True,
                                         allow_unused=True)[0]
                trace = (hv * probe).detach().abs().mean() if hv is not None else None
            except RuntimeError:
                trace = None
            if trace is None or not torch.isfinite(trace):
                trace = grad.detach().square().mean()
            old = self.running.get(name)
            self.running[name] = trace if old is None else self.ema * old + (1 - self.ema) * trace
            result[name] = self.running[name]
        return result


class HMQATController:
    """Scale latent LSQ gradients using joint Hessian/parameter sensitivity.

    Call ``update(loss, model)`` after ``loss.backward()`` and before the
    optimizer step. Fixed W4A4 is retained; no mixed-precision search is done.
    """
    source_status = "HMQAT-derived fixed-precision adaptation"

    def __init__(self, sensitivity_power=0.5, max_multiplier=4.0, update_every=1):
        self.estimator = HessianSensitivityEstimator()
        self.sensitivity_power = float(sensitivity_power)
        self.max_multiplier = float(max_multiplier)
        self.update_every = max(int(update_every), 1)
        self.step_count = 0
        self.last_sensitivity: Dict[str, float] = {}

    def update(self, loss: torch.Tensor, model: nn.Module):
        self.step_count += 1
        if self.step_count % self.update_every:
            return self.last_sensitivity
        estimates = self.estimator.estimate(loss, model.named_parameters())
        if not estimates:
            return self.last_sensitivity
        named = dict(model.named_parameters())
        raw = {n: v * named[n].detach().numel() for n, v in estimates.items()}
        ref = next(model.parameters()).device
        mean = torch.stack([v.to(ref) for v in raw.values()]).mean().clamp_min(1e-12)
        for name, value in raw.items():
            ratio = (value / mean).pow(self.sensitivity_power).clamp(
                1 / self.max_multiplier, self.max_multiplier)
            p = named[name]
            if p.grad is not None:
                p.grad.mul_(ratio.rsqrt().to(p.grad.device))
            self.last_sensitivity[name] = float(ratio.detach().cpu())
        return self.last_sensitivity

    def state_dict(self):
        return {"step_count": self.step_count, "last_sensitivity": self.last_sensitivity,
                "running": {k: v.cpu() for k, v in self.estimator.running.items()}}
