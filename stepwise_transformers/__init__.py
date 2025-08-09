"""Stepwise Transformers: Interactive transformer learning with ClearML integration.

This package provides a complete transformer implementation with comprehensive
experiment tracking and visualization for educational purposes.
"""

__version__ = "0.1.0"
__author__ = "Stepwise Transformers"

# Core model components
from stepwise_transformers.attention import ScaledDotProductAttention, MultiHeadAttention
from stepwise_transformers.models import Transformer

__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "Transformer",
]
