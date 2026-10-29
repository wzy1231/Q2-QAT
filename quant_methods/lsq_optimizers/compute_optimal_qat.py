# -*- coding: utf-8 -*-
"""Compute-optimal QAT and cooldown-QAT fusion schedule."""
from __future__ import annotations
import math


class ComputeOptimalQATSchedule:
    """Plan FP/QAT allocation and apply LR cooldown after QAT starts."""
    source_status = "Compute-Optimal QAT / cooldown-QAT adaptation"

    def __init__(self, total_epochs, qat_fraction=0.5, qat_warmup_epochs=1, min_lr_ratio=0.01):
        self.total_epochs = max(int(total_epochs), 1)
        self.qat_fraction = float(qat_fraction)
        self.qat_epochs = max(1, int(round(self.total_epochs * self.qat_fraction)))
        self.switch_epoch = self.total_epochs - self.qat_epochs
        self.qat_warmup_epochs = int(qat_warmup_epochs)
        self.min_lr_ratio = float(min_lr_ratio)

    def state_at(self, epoch):
        if int(epoch) < self.switch_epoch:
            return "fp32"
        if int(epoch) < self.switch_epoch + self.qat_warmup_epochs:
            return "qat_warmup"
        return "qat_cooldown"

    def apply(self, model, epoch):
        state = self.state_at(epoch)
        for module in model.modules():
            if hasattr(module, "enabled"):
                module.enabled = state != "fp32"
        return state

    def lr_multiplier(self, epoch):
        if int(epoch) < self.switch_epoch:
            return 1.0
        progress = min(max((int(epoch) - self.switch_epoch) / max(self.qat_epochs - 1, 1), 0.0), 1.0)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def recommendation(self, tokens_per_parameter_byte=None):
        if tokens_per_parameter_byte is None:
            return {"qat_fraction": self.qat_fraction, "qat_epochs": self.qat_epochs}
        fraction = min(max(0.25 + 0.05 * math.log10(max(tokens_per_parameter_byte, 1)), 0.25), 0.75)
        return {"qat_fraction": fraction, "qat_epochs": max(1, int(round(self.total_epochs * fraction)))}
