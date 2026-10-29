# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def ste_round(x):
    return RoundSTE.apply(x)


def reshape_channel_scale(scale, x, channel_dim):
    channel_dim = channel_dim if channel_dim >= 0 else x.ndim + channel_dim
    shape = [1] * x.ndim
    shape[channel_dim] = -1
    return scale.reshape(shape)


class QuantMethod(nn.Module):
    source_status = "paper-derived implementation"

    def enable(self, enabled=True):
        for module in self.modules():
            if hasattr(module, "enabled"):
                module.enabled = bool(enabled)
        return self
