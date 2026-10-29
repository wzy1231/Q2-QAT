from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class QGBFusion(nn.Module):
    """Gradient-balanced feature fusion for BCHW feature maps."""

    def __init__(
        self,
        dimension: int = 1,
        ema_beta: float = 0.01,
        eta: float = 0.05,
        tau: Sequence[float] = (0.0, 0.0),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d = int(dimension)
        self.ema_beta = float(ema_beta)
        self.eta = float(eta)
        self.eps = float(eps)

        self.register_buffer("tau", torch.as_tensor(tuple(tau), dtype=torch.float32))
        self.register_buffer("Gbar", torch.empty(0, dtype=torch.float32))
        self.register_buffer("lambda_logits", torch.empty(0, dtype=torch.float32))
        self.ln = None
        self._g_cache = []

    def _initialize_state(self, branch_count: int, device: torch.device) -> None:
        if self.tau.numel() != branch_count:
            raise ValueError(
                f"tau must contain one target offset per input branch; "
                f"got {self.tau.numel()} values for {branch_count} branches"
            )
        if self.Gbar.numel() != branch_count:
            self.Gbar = torch.zeros(branch_count, device=device, dtype=torch.float32)
            self.lambda_logits = torch.zeros(branch_count, device=device, dtype=torch.float32)
        else:
            self.Gbar = self.Gbar.to(device=device)
            self.lambda_logits = self.lambda_logits.to(device=device)
        self.tau = self.tau.to(device=device)

    def _build_ln_if_needed(self, fused_channels: int, device, dtype) -> None:
        if self.ln is None or tuple(self.ln.normalized_shape) != (fused_channels,):
            self.ln = nn.LayerNorm(
                fused_channels,
                elementwise_affine=False,
                eps=1e-5,
            ).to(device=device, dtype=dtype)

    def _make_grad_hook(self, index: int, feature: torch.Tensor):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad_rms = grad.detach().float().square().mean().sqrt()
            feature_rms = feature.detach().float().square().mean().sqrt()
            self._g_cache[index] = grad_rms / (feature_rms + self.eps)
            return grad

        return hook

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError("QGBFusion expects a list or tuple of feature branches")
        if any(not isinstance(feature, torch.Tensor) or feature.ndim != 4 for feature in x):
            raise ValueError("All input branches must be BCHW tensors")

        batch_size = x[0].shape[0]
        spatial_size = x[0].shape[2:]
        if any(feature.shape[0] != batch_size or feature.shape[2:] != spatial_size for feature in x):
            raise ValueError("All input branches must have matching batch and spatial dimensions")

        self._initialize_state(len(x), x[0].device)
        self._g_cache = [None] * len(x)
        if self.training:
            for index, feature in enumerate(x):
                if feature.requires_grad:
                    feature.register_hook(self._make_grad_hook(index, feature))

        alpha = F.softmax(self.lambda_logits, dim=0)
        fused = torch.cat([weight * feature for weight, feature in zip(alpha, x)], dim=self.d)
        self._build_ln_if_needed(fused.shape[1], fused.device, fused.dtype)
        return self.ln(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def closed_loop_update(self) -> None:
        if not self._g_cache or any(value is None for value in self._g_cache):
            return

        gradients = torch.stack([value.to(self.Gbar.device) for value in self._g_cache]).float()
        self.Gbar.mul_(1.0 - self.ema_beta).add_(self.ema_beta * gradients)
        log_gradients = torch.log(self.Gbar + self.eps)
        error = log_gradients - log_gradients.mean() - self.tau.to(log_gradients.dtype)
        self.lambda_logits.sub_(self.eta * error)
        self._g_cache = [None] * len(self._g_cache)

    def extra_repr(self) -> str:
        return (
            f"dimension={self.d}, ema_beta={self.ema_beta}, "
            f"eta={self.eta}, tau={tuple(self.tau.tolist())}, eps={self.eps}"
        )
