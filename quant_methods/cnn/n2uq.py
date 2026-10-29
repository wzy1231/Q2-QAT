# -*- coding: utf-8 -*-
"""N2UQ: learnable nonuniform input thresholds with uniform output levels."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import QuantMethod


class N2UQActivation(QuantMethod):
    source_status = "paper-faithful threshold/G-STE core; entropy regularization is optional"

    def __init__(self, bits=4, init_range=2.0, signed=False):
        super().__init__()
        self.bits, self.signed = int(bits), bool(signed)
        self.levels = 2 ** bits
        self.interval = init_range / max(self.levels - 1, 1)
        self.start = nn.Parameter(torch.tensor(-init_range / 2 if signed else 0.0))
        self.delta = nn.Parameter(torch.full((self.levels - 1,), self.interval))
        self.register_buffer("eps", torch.tensor(1e-4)); self.enabled = True

    def thresholds(self):
        return self.start + torch.cumsum(self.delta.clamp_min(self.eps), dim=0)

    def forward(self, x):
        if not self.enabled: return x
        thresholds = self.thresholds()
        idx = torch.bucketize(x, thresholds).clamp(0, self.levels - 1)
        hard = idx.to(x.dtype) * self.interval
        left = torch.cat([self.start.reshape(1), thresholds[:-1]])
        lower = left[idx.clamp_max(self.levels - 2)]
        width = (thresholds[idx.clamp_max(self.levels - 2)] - lower).clamp_min(self.eps)
        surrogate = ((x - lower) / width).clamp(0, 1) * self.interval
        return hard.detach() + surrogate - surrogate.detach()

    def entropy_regularization(self, weight=1e-4):
        return float(weight) * self.delta.clamp_min(self.eps).log().square().mean()


class N2UQConv2d(QuantMethod):
    def __init__(self, conv: nn.Conv2d, bits=4):
        super().__init__(); self.conv = conv; self.activation = N2UQActivation(bits)
    def forward(self, x): return self.activation(self.conv(x))


class N2UQLinear(QuantMethod):
    def __init__(self, linear: nn.Linear, bits=4):
        super().__init__(); self.linear = linear; self.activation = N2UQActivation(bits)
    def forward(self, x): return self.activation(self.linear(x))
