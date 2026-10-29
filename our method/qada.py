"""Quantization-aware Attention Distribution Alignment (Q-ADA).

The frozen full-precision teacher and quantized student provide aligned feature
maps. Their absolute difference is used to estimate quantization distortion.
Attention distributions are normalized over spatial positions per channel and
aligned using Jensen-Shannon divergence.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QADA(nn.Module):
    """Parameter-free spatial attention alignment for BCHW feature maps."""

    def __init__(self, kappa: float = 1e-5, gamma: float = 1.0):
        super().__init__()
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.kappa = float(kappa)
        self.gamma = float(gamma)

    def attention(self, feature: Tensor, delta: Tensor) -> Tensor:
        """Compute quantization-aware attention for each spatial position."""
        mu = feature.mean(dim=(-2, -1), keepdim=True)
        sigma = (feature - mu).square().mean(dim=(-2, -1), keepdim=True).sqrt()
        z = (feature - mu).abs() / (sigma + self.kappa)
        r = delta / (sigma + self.kappa)
        score = torch.log1p(z.square()) + self.gamma * torch.log1p(r.square())
        return torch.sigmoid(score)

    def spatial_distribution(self, feature: Tensor, delta: Tensor) -> Tensor:
        """Normalize attention over H and W separately for each channel."""
        attention = self.attention(feature, delta)
        return attention / attention.sum(dim=(-2, -1), keepdim=True)

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        """Return the batch- and channel-averaged spatial JS divergence.

        Args:
            student: Quantized student features of shape [B, C, H, W].
            teacher: Frozen full-precision teacher features of the same shape.
        """
        if student.ndim != 4 or teacher.shape != student.shape:
            raise ValueError("Student and teacher must have matching BCHW shapes")
        if student.device != teacher.device:
            raise ValueError("Student and teacher must be on the same device")

        student = student.float()
        teacher = teacher.detach().float()
        delta = (teacher - student).abs()

        with torch.no_grad():
            p = self.spatial_distribution(teacher, delta.detach())
        q = self.spatial_distribution(student, delta)
        m = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.log() - m.log())).sum(dim=(-2, -1))
            + (q * (q.log() - m.log())).sum(dim=(-2, -1))
        )
        return js.mean()

    def js_divergence(self, student: Tensor, teacher: Tensor) -> Tensor:
        """Compute the Q-ADA alignment loss."""
        return self.forward(student, teacher)


if __name__ == "__main__":
    student = torch.randn(2, 8, 16, 16, requires_grad=True)
    teacher = torch.randn_like(student)
    loss = QADA()(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    print(f"Q-ADA JS loss: {loss.item():.6f}")
