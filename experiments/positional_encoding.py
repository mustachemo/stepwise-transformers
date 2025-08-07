"""Positional encoding modules for transformer learning.

This module provides components for learning about positional encoding
in transformers through interactive MLflow tracking.
"""

from __future__ import annotations

import mlflow
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console

from .base import BaseModule


class PositionalEncodingModule(BaseModule):
    """Interactive positional encoding components."""

    def __init__(self, manager):
        """Initialize the positional encoding module.

        Args:
            manager: The main experiment manager instance.
        """
        super().__init__(manager)
        self.console = Console()

    def sinusoidal_encoding(self, seq_length: int, d_model: int) -> torch.Tensor:
        """Compute sinusoidal positional encoding.

        Args:
            seq_length: Length of the sequence.
            d_model: Model dimension.

        Returns:
            torch.Tensor: Positional encoding tensor.

        Raises:
            RuntimeError: If encoding computation fails.
        """
        try:
            with mlflow.start_run(run_name="sinusoidal_positional_encoding") as run:
                # Log parameters
                mlflow.log_param("seq_length", seq_length)
                mlflow.log_param("d_model", d_model)

                # Create positional encoding
                pe = torch.zeros(seq_length, d_model)
                position = torch.arange(0, seq_length).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   -(torch.log(torch.tensor(10000.0)) / d_model))

                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)

                # Log metrics
                mlflow.log_metric("encoding_norm", torch.norm(pe))
                mlflow.log_metric("encoding_std", torch.std(pe))

                logger.info("Sinusoidal positional encoding computation completed")
                return pe

        except Exception as exc:
            logger.error(f"Sinusoidal positional encoding computation failed: {exc}")
            raise RuntimeError(f"Positional encoding computation failed: {exc}")

    def learned_encoding(self, seq_length: int, d_model: int) -> torch.Tensor:
        """Compute learned positional encoding.

        Args:
            seq_length: Length of the sequence.
            d_model: Model dimension.

        Returns:
            torch.Tensor: Learned positional encoding tensor.

        Raises:
            RuntimeError: If encoding computation fails.
        """
        try:
            with mlflow.start_run(run_name="learned_positional_encoding") as run:
                # Log parameters
                mlflow.log_param("seq_length", seq_length)
                mlflow.log_param("d_model", d_model)

                # Create learned positional encoding
                pe = torch.randn(seq_length, d_model) * 0.1
                pe.requires_grad_(True)

                # Log metrics
                mlflow.log_metric("encoding_norm", torch.norm(pe))
                mlflow.log_metric("encoding_std", torch.std(pe))

                logger.info("Learned positional encoding computation completed")
                return pe

        except Exception as exc:
            logger.error(f"Learned positional encoding computation failed: {exc}")
            raise RuntimeError(f"Learned positional encoding computation failed: {exc}")

    def relative_encoding(self, seq_length: int, d_model: int, max_relative_position: int = 32) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            seq_length: Length of the sequence.
            d_model: Model dimension.
            max_relative_position: Maximum relative position to encode.

        Returns:
            torch.Tensor: Relative positional encoding tensor.

        Raises:
            RuntimeError: If encoding computation fails.
        """
        try:
            with mlflow.start_run(run_name="relative_positional_encoding") as run:
                # Log parameters
                mlflow.log_param("seq_length", seq_length)
                mlflow.log_param("d_model", d_model)
                mlflow.log_param("max_relative_position", max_relative_position)

                # Create relative positional encoding
                vocab_size = 2 * max_relative_position + 1
                pe = torch.randn(vocab_size, d_model) * 0.1

                # Log metrics
                mlflow.log_metric("vocab_size", vocab_size)
                mlflow.log_metric("encoding_norm", torch.norm(pe))
                mlflow.log_metric("encoding_std", torch.std(pe))

                logger.info("Relative positional encoding computation completed")
                return pe

        except Exception as exc:
            logger.error(f"Relative positional encoding computation failed: {exc}")
            raise RuntimeError(f"Relative positional encoding computation failed: {exc}")

    def forward(self, seq_length: int, d_model: int, encoding_type: str = "sinusoidal") -> torch.Tensor:
        """Forward pass through the positional encoding module.

        Args:
            seq_length: Length of the sequence.
            d_model: Model dimension.
            encoding_type: Type of encoding to use ('sinusoidal', 'learned', 'relative').

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        if encoding_type == "sinusoidal":
            return self.sinusoidal_encoding(seq_length, d_model)
        elif encoding_type == "learned":
            return self.learned_encoding(seq_length, d_model)
        elif encoding_type == "relative":
            return self.relative_encoding(seq_length, d_model)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
