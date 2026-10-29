# -*- coding: utf-8 -*-
"""Q-MHSA core: cross-head concordance and learnable smoothness control."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import QuantMethod


class CrossHeadConcordance(nn.Module):
    def __init__(self, num_heads, channels_per_head):
        super().__init__(); self.gate = nn.Parameter(torch.zeros(num_heads, channels_per_head))
    def forward(self, value):
        mean = value.mean(dim=1, keepdim=True)
        return value + torch.tanh(self.gate).view(1, -1, 1, self.gate.shape[-1]) * (mean - value)


class LearnableSmoothnessController(nn.Module):
    def __init__(self, num_heads, init=1.0):
        super().__init__(); self.log_temperature = nn.Parameter(torch.full((num_heads, 1, 1), math.log(init)))
    def forward(self, logits):
        temperature = self.log_temperature.exp().clamp(0.1, 10.0)
        return logits / temperature


class QMHSA(QuantMethod):
    source_status = "paper-derived CCM/LSC core; quantizer integration is framework-agnostic"

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(); assert dim % num_heads == 0
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim); self.drop = nn.Dropout(dropout)
        self.ccm = CrossHeadConcordance(num_heads, self.head_dim); self.lsc = LearnableSmoothnessController(num_heads); self.enabled = True

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], self.ccm(qkv[2])
        logits = self.lsc(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim))
        attn = self.drop(logits.softmax(dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, self.dim))
