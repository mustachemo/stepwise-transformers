"""Transformer block modules for transformer learning.

This module provides components for learning about transformer blocks
through interactive MLflow tracking.
"""

from __future__ import annotations

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from rich.console import Console

from .base import BaseModule


class TransformerBlockModule(BaseModule):
    """Interactive transformer block components."""

    def __init__(self, manager):
        """Initialize the transformer block module.

        Args:
            manager: The main experiment manager instance.
        """
        super().__init__(manager)
        self.console = Console()

    def encoder_block(
        self,
        input_tensor: torch.Tensor,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> torch.Tensor:
        """Compute transformer encoder block.

        Args:
            input_tensor: Input tensor.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout rate.

        Returns:
            torch.Tensor: Encoder block output.

        Raises:
            RuntimeError: If encoder block computation fails.
        """
        try:
            with mlflow.start_run(run_name="transformer_encoder_block") as run:
                # Log parameters
                mlflow.log_param("d_model", d_model)
                mlflow.log_param("num_heads", num_heads)
                mlflow.log_param("d_ff", d_ff)
                mlflow.log_param("dropout", dropout)
                mlflow.log_param("input_shape", list(input_tensor.shape))

                # Multi-head attention
                attention_output = self._multi_head_attention(
                    input_tensor, d_model, num_heads, dropout
                )

                # Add & Norm
                residual_output = input_tensor + attention_output
                normalized_output = self._layer_norm(residual_output, d_model)

                # Feed-forward network
                ff_output = self._feed_forward(
                    normalized_output, d_model, d_ff, dropout
                )

                # Add & Norm
                final_output = normalized_output + ff_output
                output = self._layer_norm(final_output, d_model)

                # Log metrics
                mlflow.log_metric("output_norm", torch.norm(output))
                mlflow.log_metric("residual_norm", torch.norm(residual_output))

                logger.info("Transformer encoder block computation completed")
                return output

        except Exception as exc:
            logger.error(f"Transformer encoder block computation failed: {exc}")
            raise RuntimeError(f"Encoder block computation failed: {exc}")

    def decoder_block(
        self,
        input_tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> torch.Tensor:
        """Compute transformer decoder block.

        Args:
            input_tensor: Input tensor.
            encoder_output: Output from encoder.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout rate.

        Returns:
            torch.Tensor: Decoder block output.

        Raises:
            RuntimeError: If decoder block computation fails.
        """
        try:
            with mlflow.start_run(run_name="transformer_decoder_block") as run:
                # Log parameters
                mlflow.log_param("d_model", d_model)
                mlflow.log_param("num_heads", num_heads)
                mlflow.log_param("d_ff", d_ff)
                mlflow.log_param("dropout", dropout)
                mlflow.log_param("input_shape", list(input_tensor.shape))
                mlflow.log_param("encoder_output_shape", list(encoder_output.shape))

                # Self-attention with masking
                self_attention_output = self._masked_multi_head_attention(
                    input_tensor, d_model, num_heads, dropout
                )

                # Add & Norm
                residual_output = input_tensor + self_attention_output
                normalized_output = self._layer_norm(residual_output, d_model)

                # Cross-attention
                cross_attention_output = self._cross_attention(
                    normalized_output, encoder_output, d_model, num_heads, dropout
                )

                # Add & Norm
                residual_output = normalized_output + cross_attention_output
                normalized_output = self._layer_norm(residual_output, d_model)

                # Feed-forward network
                ff_output = self._feed_forward(
                    normalized_output, d_model, d_ff, dropout
                )

                # Add & Norm
                final_output = normalized_output + ff_output
                output = self._layer_norm(final_output, d_model)

                # Log metrics
                mlflow.log_metric("output_norm", torch.norm(output))
                mlflow.log_metric("residual_norm", torch.norm(residual_output))

                logger.info("Transformer decoder block computation completed")
                return output

        except Exception as exc:
            logger.error(f"Transformer decoder block computation failed: {exc}")
            raise RuntimeError(f"Decoder block computation failed: {exc}")

    def _multi_head_attention(
        self, x: torch.Tensor, d_model: int, num_heads: int, dropout: float
    ) -> torch.Tensor:
        """Compute multi-head attention."""
        # Implementation details
        d_k = d_model // num_heads
        attention_scores = torch.matmul(x, x.transpose(-2, -1)) / (d_k**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, dropout)
        return torch.matmul(attention_weights, x)

    def _masked_multi_head_attention(
        self, x: torch.Tensor, d_model: int, num_heads: int, dropout: float
    ) -> torch.Tensor:
        """Compute masked multi-head attention."""
        # Implementation details
        d_k = d_model // num_heads
        attention_scores = torch.matmul(x, x.transpose(-2, -1)) / (d_k**0.5)

        # Apply causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(mask, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, dropout)
        return torch.matmul(attention_weights, x)

    def _cross_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> torch.Tensor:
        """Compute cross-attention."""
        # Implementation details
        d_k = d_model // num_heads
        attention_scores = torch.matmul(query, key_value.transpose(-2, -1)) / (d_k**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, dropout)
        return torch.matmul(attention_weights, key_value)

    def _feed_forward(
        self, x: torch.Tensor, d_model: int, d_ff: int, dropout: float
    ) -> torch.Tensor:
        """Compute feed-forward network."""
        # Implementation details
        return F.dropout(F.relu(F.linear(x, torch.randn(d_ff, d_model))), dropout)

    def _layer_norm(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
        """Compute layer normalization."""
        # Implementation details
        return F.layer_norm(x, [d_model])

    def forward(
        self,
        input_tensor: torch.Tensor,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        block_type: str = "encoder",
    ) -> torch.Tensor:
        """Forward pass through the transformer block module.

        Args:
            input_tensor: Input tensor.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout rate.
            block_type: Type of block ('encoder' or 'decoder').

        Returns:
            torch.Tensor: Transformer block output.
        """
        if block_type == "encoder":
            return self.encoder_block(input_tensor, d_model, num_heads, d_ff, dropout)
        elif block_type == "decoder":
            # For decoder, we need encoder output as well
            encoder_output = torch.randn_like(input_tensor)  # Placeholder
            return self.decoder_block(
                input_tensor, encoder_output, d_model, num_heads, d_ff, dropout
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")
