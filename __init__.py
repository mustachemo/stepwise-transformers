"""Stepwise Transformers Learning with MLflow.

This package provides an interactive learning environment for transformer architecture,
attention mechanisms, and neural network concepts through MLflow tracking.
"""

__version__ = "0.1.0"
__author__ = "Stepwise Transformers Team"

from .transformer_manager import TransformerManager
from .components.attention import AttentionModule
from .components.positional_encoding import PositionalEncodingModule
from .components.transformer_block import TransformerBlockModule

__all__ = [
    "TransformerManager",
    "AttentionModule",
    "PositionalEncodingModule",
    "TransformerBlockModule",
]
