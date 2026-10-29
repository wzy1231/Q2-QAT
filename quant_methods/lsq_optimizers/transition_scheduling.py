# -*- coding: utf-8 -*-
"""Transition-rate scheduling (TR) for LSQ latent weights."""
from __future__ import annotations
import torch


def _iter_quantizers(model):
    for module in model.modules():
        if hasattr(module, "scale") and hasattr(module, "qmax") and hasattr(module, "qmin"):
            yield module


class TransitionRateScheduler:
    """Adapt the optimizer LR to a scheduled rate of discrete LSQ transitions."""
    source_status = "TR-derived transition-adaptive LSQ optimizer"

    def __init__(self, optimizer, total_steps, start_rate=0.10, end_rate=0.005,
                 warmup_steps=0, lr_min_factor=0.1, lr_max_factor=4.0):
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.start_rate, self.end_rate = float(start_rate), float(end_rate)
        self.warmup_steps = int(warmup_steps)
        self.lr_min_factor, self.lr_max_factor = float(lr_min_factor), float(lr_max_factor)
        self.step_count, self._last_states = 0, None
        self.last_transition_rate = 0.0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    @torch.no_grad()
    def before_step(self, model):
        states = []
        for q in _iter_quantizers(model):
            latent = getattr(q, "_latent_for_transition", None)
            if latent is not None:
                states.append(torch.round(latent.detach() / q.scale.detach().abs().clamp_min(1e-8)).cpu())
        self._last_states = states

    @torch.no_grad()
    def after_step(self, model):
        current = []
        for q in _iter_quantizers(model):
            latent = getattr(q, "_latent_for_transition", None)
            if latent is not None:
                current.append(torch.round(latent.detach() / q.scale.detach().abs().clamp_min(1e-8)).cpu())
        if self._last_states and current:
            changed = total = 0
            for old, new in zip(self._last_states, current):
                n = min(old.numel(), new.numel())
                changed += int((old.reshape(-1)[:n] != new.reshape(-1)[:n]).sum())
                total += n
            self.last_transition_rate = changed / max(total, 1)
        self.step_count += 1
        target = self.start_rate + (self.end_rate - self.start_rate) * min(
            self.step_count / self.total_steps, 1.0)
        factor = (1.0 + 2.0 * (target - self.last_transition_rate))
        factor = min(max(factor, self.lr_min_factor), self.lr_max_factor)
        if self.step_count <= self.warmup_steps:
            factor = min(factor, 1.0)
        for base, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base * factor
        return self.last_transition_rate

    def step(self, model):
        self.before_step(model)
        self.optimizer.step()
        return self.after_step(model)
