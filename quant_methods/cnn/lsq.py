# -*- coding: utf-8 -*-
"""LSQ: Learned Step Size Quantization (ICLR 2020)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import QuantMethod, RoundSTE, reshape_channel_scale


def _grad_scale(value, factor):
    return (value - value.detach()) * factor + value.detach()


class LSQFakeQuantizer(QuantMethod):
    source_status = "paper-faithful core; layer replacement is a local adapter"

    def __init__(self, bits=4, signed=True, channel_dim=None, channels=None):
        super().__init__()
        self.bits, self.signed, self.channel_dim = int(bits), bool(signed), channel_dim
        self.qmin = -(2 ** (bits - 1)) if signed else 0
        self.qmax = 2 ** (bits - 1) - 1 if signed else 2 ** bits - 1
        size = int(channels) if channel_dim is not None and channels else 1
        self.scale = nn.Parameter(torch.ones(size))
        self.register_buffer("initialized", torch.tensor(False))
        self.enabled = True

    @torch.no_grad()
    def initialize(self, x):
        if bool(self.initialized):
            return
        dims = tuple(i for i in range(x.ndim) if i != self.channel_dim) if self.channel_dim is not None else tuple(range(x.ndim))
        mean_abs = x.detach().abs().mean(dim=dims)
        init = 2.0 * mean_abs / max(self.qmax ** 0.5, 1.0)
        self.scale.copy_(init.clamp_min(1e-8).reshape_as(self.scale))
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.enabled:
            return x
        self.initialize(x)
        scale = self.scale.abs().clamp_min(1e-8)
        if self.channel_dim is not None:
            scale = reshape_channel_scale(scale, x, self.channel_dim)
            elements_per_scale = x.numel() / max(self.scale.numel(), 1)
        else:
            elements_per_scale = x.numel()
        scale = _grad_scale(scale, 1.0 / max((elements_per_scale * max(self.qmax, 1)) ** 0.5, 1.0))
        y = x / scale
        q = torch.clamp(RoundSTE.apply(y), self.qmin, self.qmax)
        return q * scale


class LSQConv2d(QuantMethod):
    def __init__(self, conv: nn.Conv2d, bits=4, quantize_weight=True, quantize_activation=True):
        super().__init__()
        self.conv = conv
        self.wq = LSQFakeQuantizer(bits, True, 0, conv.out_channels) if quantize_weight else nn.Identity()
        self.aq = LSQFakeQuantizer(bits, True, 1, conv.out_channels) if quantize_activation else nn.Identity()

    def forward(self, x):
        # Expose the latent weight to transition-aware LSQ schedulers without
        # changing the forward computation or registering a second parameter.
        if hasattr(self.wq, "scale"):
            self.wq._latent_for_transition = self.conv.weight
        y = F.conv2d(x, self.wq(self.conv.weight), self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return self.aq(y)


class LSQLinear(QuantMethod):
    def __init__(self, linear: nn.Linear, bits=4, quantize_weight=True, quantize_activation=True):
        super().__init__()
        self.linear = linear
        self.wq = LSQFakeQuantizer(bits, True, 0, linear.out_features) if quantize_weight else nn.Identity()
        self.aq = LSQFakeQuantizer(bits, True, -1, linear.out_features) if quantize_activation else nn.Identity()

    def forward(self, x):
        if hasattr(self.wq, "scale"):
            self.wq._latent_for_transition = self.linear.weight
        return self.aq(F.linear(x, self.wq(self.linear.weight), self.linear.bias))
