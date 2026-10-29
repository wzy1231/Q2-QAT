# -*- coding: utf-8 -*-
"""QT-DoG-style staged quantization schedule for an LSQ YOLOv5 baseline."""
from __future__ import annotations


class QTDoGSchedule:
    """FP32 -> activation-only -> full W4A4 schedule from the supplied protocol."""
    source_status = "QT-DoG adaptation for fixed-budget YOLOv5 QAT"

    def __init__(self, fp_epochs=10, activation_epochs=10, total_epochs=100):
        self.fp_epochs, self.activation_epochs = int(fp_epochs), int(activation_epochs)
        self.total_epochs, self.stage = int(total_epochs), -1

    def stage_at(self, epoch):
        if int(epoch) < self.fp_epochs:
            return 0
        if int(epoch) < self.fp_epochs + self.activation_epochs:
            return 1
        return 2

    def apply(self, model, epoch):
        self.stage = self.stage_at(epoch)
        for module in model.modules():
            if hasattr(module, "enabled"):
                module.enabled = self.stage > 0
            if hasattr(module, "quantize_weight"):
                module.quantize_weight = self.stage >= 2
            if hasattr(module, "quantize_activation"):
                module.quantize_activation = self.stage >= 1
        return self.stage

    def describe(self, epoch):
        return ("fp32" if self.stage_at(epoch) == 0 else
                "activation-only" if self.stage_at(epoch) == 1 else "full-W4A4")
