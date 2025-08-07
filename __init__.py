"""Stepwise Transformers Learning with MLflow.

This package provides an interactive learning environment for transformer architecture,
attention mechanisms, and neural network concepts through MLflow tracking.
"""

__version__ = "0.1.0"
__author__ = "Stepwise Transformers Team"

from .experiment_manager import TransformerExperimentManager
from .experiments.attention import AttentionModule
from .experiments.positional_encoding import PositionalEncodingModule
from .experiments.transformer_block import TransformerBlockModule

__all__ = [
    "TransformerExperimentManager",
    "AttentionModule",
    "PositionalEncodingModule", 
    "TransformerBlockModule",
]
