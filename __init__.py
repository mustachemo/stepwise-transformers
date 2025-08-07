"""Stepwise Transformers Learning with MLflow.

This package provides an interactive learning environment for transformer architecture,
attention mechanisms, and neural network concepts through MLflow experiments.
"""

__version__ = "0.1.0"
__author__ = "Stepwise Transformers Team"

from .experiment_manager import TransformerExperimentManager
from .experiments.attention import AttentionExperiment
from .experiments.positional_encoding import PositionalEncodingExperiment
from .experiments.transformer_block import TransformerBlockExperiment

__all__ = [
    "TransformerExperimentManager",
    "AttentionExperiment",
    "PositionalEncodingExperiment",
    "TransformerBlockExperiment",
]
