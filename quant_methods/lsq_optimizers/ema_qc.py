# -*- coding: utf-8 -*-
"""EMA and quantization correction for LSQ-based QAT."""
from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn


class EMALSQController:
    """Maintain EMA copies of latent model parameters and LSQ scale factors."""
    source_status = "EMA/QC-derived YOLO QAT adaptation"

    def __init__(self, decay=0.999, warmup=True):
        self.decay, self.warmup = float(decay), bool(warmup)
        self.num_updates, self.shadow = 0, OrderedDict()

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates)) if self.warmup else self.decay
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                if name not in self.shadow:
                    self.shadow[name] = value.detach().clone()
                else:
                    self.shadow[name].mul_(d).add_(value.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": dict(self.shadow)}


class QuantizationCorrection(nn.Module):
    """Fit a small per-channel affine correction on a calibration pair."""
    def __init__(self, channels=1, momentum=0.01):
        super().__init__()
        self.channels, self.momentum = int(channels), float(momentum)
        self.scale = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x):
        shape = [1] * x.ndim
        shape[1 if x.ndim > 1 else 0] = self.channels
        return x * self.scale.reshape(shape) + self.bias.reshape(shape)

    @torch.no_grad()
    def fit(self, quantized, reference):
        q, r = quantized.detach(), reference.detach()
        dim = tuple(i for i in range(q.ndim) if i != 1)
        qmean, rmean = q.mean(dim), r.mean(dim)
        qstd, rstd = q.std(dim).clamp_min(1e-6), r.std(dim).clamp_min(1e-6)
        self.scale.data.lerp_(rstd / qstd, self.momentum)
        self.bias.data.lerp_(rmean - self.scale.detach() * qmean, self.momentum)
        return self
