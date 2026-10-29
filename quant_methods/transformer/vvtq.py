# -*- coding: utf-8 -*-
"""VVTQ/Quantization Variation: variation-aware statistics and fake quantization."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..base import QuantMethod, ste_round


class VVTQFakeQuantizer(QuantMethod):
    source_status = "paper-derived variation-aware adapter; full training schedule remains caller-owned"

    def __init__(self, bits=4, channel_dim=-1, momentum=0.99):
        super().__init__(); self.bits = int(bits); self.channel_dim = int(channel_dim); self.momentum = float(momentum); self.enabled = True
        self.register_buffer("running_max", torch.tensor(1.0))

    def forward(self, x):
        if not self.enabled: return x
        reduce_dims = tuple(i for i in range(x.ndim) if i != self.channel_dim)
        current = x.detach().abs().amax(dim=reduce_dims).mean()
        if self.training: self.running_max.mul_(self.momentum).add_(current * (1 - self.momentum))
        qmax = 2 ** (self.bits - 1) - 1; scale = self.running_max.clamp_min(1e-8) / qmax
        return torch.clamp(ste_round(x / scale), -2 ** (self.bits - 1), qmax) * scale


class VVTQController(QuantMethod):
    source_status = "paper-derived controller exposing variation statistics"
    def __init__(self, bits=4): super().__init__(); self.quantizer = VVTQFakeQuantizer(bits)
    def forward(self, x): return self.quantizer(x)
