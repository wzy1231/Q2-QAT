"""Pluggable quantization methods for CNNs and Vision Transformers."""

from .registry import available_methods, build_quantizer

__all__ = ["available_methods", "build_quantizer"]
