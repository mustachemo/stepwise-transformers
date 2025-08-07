"""Attention mechanism modules for transformer learning.

This module provides components for learning about attention mechanisms
in transformers through interactive MLflow tracking.
"""

from __future__ import annotations

from typing import Optional

import mlflow
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console

from .base import BaseModule


class AttentionModule(BaseModule):
    """Interactive attention mechanism components."""

    def __init__(self, manager):
        """Initialize the attention module.

        Args:
            manager: The main experiment manager instance.
        """
        super().__init__(manager)
        self.console = Console()

    def single_head_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Compute single-head attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.

        Returns:
            torch.Tensor: Attention output.

        Raises:
            RuntimeError: If attention computation fails.
        """
        try:
            with mlflow.start_run(run_name="single_head_attention") as run:
                # Log input parameters
                mlflow.log_param("query_shape", list(query.shape))
                mlflow.log_param("key_shape", list(key.shape))
                mlflow.log_param("value_shape", list(value.shape))

                # Compute attention
                attention_scores = torch.matmul(query, key.transpose(-2, -1))
                attention_weights = torch.softmax(attention_scores, dim=-1)
                output = torch.matmul(attention_weights, value)

                # Log attention weights visualization
                self.manager.log_attention_visualization(attention_weights, step=0)

                # Log metrics
                mlflow.log_metric(
                    "attention_entropy",
                    -torch.sum(attention_weights * torch.log(attention_weights + 1e-8)),
                )
                mlflow.log_metric("output_norm", torch.norm(output))

                logger.info("Single-head attention computation completed")
                return output

        except Exception as exc:
            logger.error(f"Single-head attention computation failed: {exc}")
            raise RuntimeError(f"Attention computation failed: {exc}")

    def multi_head_attention(
        self,
        num_heads: int,
        d_model: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            num_heads: Number of attention heads.
            d_model: Model dimension.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.

        Returns:
            torch.Tensor: Multi-head attention output.

        Raises:
            RuntimeError: If multi-head attention computation fails.
        """
        try:
            with mlflow.start_run(run_name="multi_head_attention") as run:
                # Log parameters
                mlflow.log_param("num_heads", num_heads)
                mlflow.log_param("d_model", d_model)
                mlflow.log_param("query_shape", list(query.shape))
                mlflow.log_param("key_shape", list(key.shape))
                mlflow.log_param("value_shape", list(value.shape))

                # Split into multiple heads
                d_k = d_model // num_heads
                query_heads = query.view(-1, num_heads, d_k)
                key_heads = key.view(-1, num_heads, d_k)
                value_heads = value.view(-1, num_heads, d_k)

                # Compute attention for each head
                attention_outputs = []
                for head in range(num_heads):
                    head_output = self.single_head_attention(
                        query_heads[:, head], key_heads[:, head], value_heads[:, head]
                    )
                    attention_outputs.append(head_output)

                # Concatenate head outputs
                output = torch.cat(attention_outputs, dim=-1)

                # Log multi-head metrics
                mlflow.log_metric("num_heads_used", num_heads)
                mlflow.log_metric("output_norm", torch.norm(output))

                logger.info("Multi-head attention computation completed")
                return output

        except Exception as exc:
            logger.error(f"Multi-head attention computation failed: {exc}")
            raise RuntimeError(f"Multi-head attention computation failed: {exc}")

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional attention mask.

        Returns:
            torch.Tensor: Scaled dot-product attention output.

        Raises:
            RuntimeError: If scaled dot-product attention computation fails.
        """
        try:
            with mlflow.start_run(run_name="scaled_dot_product_attention") as run:
                # Log parameters
                mlflow.log_param("query_shape", list(query.shape))
                mlflow.log_param("key_shape", list(key.shape))
                mlflow.log_param("value_shape", list(value.shape))
                mlflow.log_param("has_mask", mask is not None)

                # Compute scaled dot-product attention
                d_k = query.size(-1)
                scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)

                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)

                attention_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attention_weights, value)

                # Log metrics
                mlflow.log_metric("d_k", d_k)
                mlflow.log_metric(
                    "attention_entropy",
                    -torch.sum(attention_weights * torch.log(attention_weights + 1e-8)),
                )
                mlflow.log_metric("output_norm", torch.norm(output))

                logger.info("Scaled dot-product attention computation completed")
                return output

        except Exception as exc:
            logger.error(f"Scaled dot-product attention computation failed: {exc}")
            raise RuntimeError(
                f"Scaled dot-product attention computation failed: {exc}"
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the attention module.

        This method provides a unified interface for all attention computations.
        """
        return self.scaled_dot_product_attention(*args, **kwargs)
