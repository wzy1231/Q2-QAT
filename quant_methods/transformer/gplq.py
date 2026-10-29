# -*- coding: utf-8 -*-
"""GPLQ: activation-first QAT controller for Vision Transformers."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import QuantMethod, ste_round


class GPLQActivationQuantizer(QuantMethod):
    source_status = "paper-faithful activation-first controller; weight PTQ policy is caller-configurable"

    def __init__(self, bits=4, per_channel=False, channel_dim=-1):
        super().__init__(); self.bits = int(bits); self.per_channel = per_channel; self.channel_dim = channel_dim; self.enabled = True
        self.scale = nn.Parameter(torch.ones(1)); self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, x):
        if bool(self.initialized): return
        dims = tuple(i for i in range(x.ndim) if i != self.channel_dim) if self.per_channel else tuple(range(x.ndim))
        maximum = x.detach().abs().amax()
        self.scale.data.fill_(float((maximum / max(2 ** (self.bits - 1) - 1, 1)).clamp_min(1e-8)))
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.enabled: return x
        self.initialize(x); scale = self.scale.abs().clamp_min(1e-8)
        if self.per_channel:
            shape = [1] * x.ndim; shape[self.channel_dim] = -1; scale = scale.reshape(shape)
        qmax = 2 ** (self.bits - 1) - 1
        return torch.clamp(ste_round(x / scale), -2 ** (self.bits - 1), qmax) * scale


class GPLQController(QuantMethod):
    source_status = "paper-faithful two-stage schedule"
    def __init__(self, activation_bits=4):
        super().__init__(); self.activation_bits = int(activation_bits); self.stage = 1
    def stage_one(self): self.stage = 1; return self
    def stage_two(self): self.stage = 2; return self
    def feature_mimic_loss(self, student, teacher): return F.mse_loss(student, teacher.detach())
