# -*- coding: utf-8 -*-
"""Core Q-DETR adapter: query-aware low-bit fake quantization and distillation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import QuantMethod, ste_round


class QDETRQuantizer(QuantMethod):
    source_status = "paper-derived Q-DETR quantizer/distillation adapter; DETR training loop remains caller-owned"

    def __init__(self, bits=4, temperature=1.0):
        super().__init__(); self.bits = int(bits); self.temperature = float(temperature); self.enabled = True

    def forward(self, x):
        if not self.enabled: return x
        qmax = 2 ** (self.bits - 1) - 1
        scale = x.detach().abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=True).clamp_min(1e-8) / qmax
        return torch.clamp(ste_round(x / scale), -qmax - 1, qmax) * scale


class QDETRQueryDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, weight=1.0):
        super().__init__(); self.temperature = float(temperature); self.weight = float(weight)

    def forward(self, student_queries, teacher_queries):
        t = self.temperature
        return self.weight * F.kl_div(F.log_softmax(student_queries / t, dim=-1), F.softmax(teacher_queries.detach() / t, dim=-1), reduction="batchmean") * (t * t)
