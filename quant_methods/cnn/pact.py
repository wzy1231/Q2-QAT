# -*- coding: utf-8 -*-
"""PACT: Parameterized Clipping Activation (CoRR 2018)."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..base import QuantMethod, ste_round


class PACTActivation(QuantMethod):
    source_status = "paper-derived; alpha regularization is exposed to the caller"

    def __init__(self, bits=4, alpha=6.0, signed=False):
        super().__init__()
        self.bits, self.signed = int(bits), bool(signed)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x
        alpha = self.alpha.abs().clamp_min(1e-6)
        if self.signed:
            qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
            scale = alpha / max(qmax, 1)
            y = torch.minimum(torch.maximum(x, -alpha), alpha) / scale
        else:
            qmin, qmax = 0, 2 ** self.bits - 1
            scale = alpha / max(qmax, 1)
            y = torch.minimum(torch.maximum(x, x.new_zeros(())), alpha) / scale
        return torch.clamp(ste_round(y), qmin, qmax) * scale

    def alpha_regularization(self, weight=1e-4):
        return float(weight) * self.alpha.square()


class PACTConv2d(QuantMethod):
    def __init__(self, conv: nn.Conv2d, bits=4, alpha=6.0):
        super().__init__(); self.conv = conv; self.activation = PACTActivation(bits, alpha)
    def forward(self, x): return self.activation(self.conv(x))


class PACTLinear(QuantMethod):
    def __init__(self, linear: nn.Linear, bits=4, alpha=6.0):
        super().__init__(); self.linear = linear; self.activation = PACTActivation(bits, alpha)
    def forward(self, x): return self.activation(self.linear(x))
