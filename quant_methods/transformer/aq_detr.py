# -*- coding: utf-8 -*-
"""AQ-DETR adapter: auxiliary queries and layer-wise feature distillation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import QuantMethod


class AQDETR(QuantMethod):
    source_status = "paper-derived auxiliary-query module; matching/criterion integration is caller-owned"

    def __init__(self, hidden_dim, num_aux_queries=100, dropout=0.0):
        super().__init__(); self.aux_queries = nn.Embedding(num_aux_queries, hidden_dim); self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size, device=None):
        device = device or self.aux_queries.weight.device
        return self.dropout(self.aux_queries.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device))


class AuxiliaryQueryDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, weight=1.0):
        super().__init__(); self.temperature = float(temperature); self.weight = float(weight)

    def forward(self, student_layers, teacher_layers):
        if len(student_layers) != len(teacher_layers):
            raise ValueError("student_layers and teacher_layers must have equal length")
        loss = student_layers[0].new_zeros(())
        for student, teacher in zip(student_layers, teacher_layers):
            loss = loss + F.mse_loss(student, teacher.detach())
        return self.weight * loss / max(len(student_layers), 1)
