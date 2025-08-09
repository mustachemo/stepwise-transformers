"""Scaled dot-product attention implementation using PyTorch.

This module provides a wrapper around PyTorch's scaled_dot_product_attention
function with additional logging and visualization capabilities for educational purposes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with educational logging and visualization.

    This implementation leverages PyTorch's optimized scaled_dot_product_attention
    function while adding comprehensive logging and visualization capabilities
    for learning purposes.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        temperature: Optional[float] = None,
    ) -> None:
        """Initialize scaled dot-product attention.

        Args:
            d_model: Model dimension (used for scaling if temperature not provided).
            dropout: Dropout probability for attention weights.
            temperature: Temperature for attention scaling. If None, uses sqrt(d_model).

        Raises:
            ValueError: If d_model is not positive or dropout is not in [0, 1].
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature or math.sqrt(d_model)

        # For storing attention weights for visualization
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_v, d_model).
            mask: Optional attention mask. True values are masked out.
            store_attention: Whether to store attention weights for visualization.

        Returns:
            Tuple of (attention_output, attention_weights).
            attention_weights is None unless store_attention=True.

        Raises:
            ValueError: If tensor dimensions are incompatible.
        """
        batch_size, seq_len_q, d_q = query.shape
        batch_size_k, seq_len_k, d_k = key.shape
        batch_size_v, seq_len_v, d_v = value.shape

        # Validate input dimensions
        if not (batch_size == batch_size_k == batch_size_v):
            raise ValueError("Batch sizes must match across query, key, and value")
        if not (d_q == d_k == d_v == self.d_model):
            raise ValueError(f"Feature dimensions must match d_model={self.d_model}")
        if seq_len_k != seq_len_v:
            raise ValueError("Key and value sequence lengths must match")

        # Use PyTorch's optimized implementation for the core computation
        if hasattr(F, "scaled_dot_product_attention") and not store_attention:
            # PyTorch 2.0+ optimized implementation
            attention_mask = None
            if mask is not None:
                # Convert boolean mask to float mask for PyTorch function
                attention_mask = mask.float().masked_fill(mask, float("-inf"))

            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0 / self.temperature,
            )
            return output, None

        # Manual implementation for educational purposes and attention visualization
        return self._manual_attention(query, key, value, mask, store_attention)

    def _manual_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        store_attention: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Manual attention computation for educational purposes.

        This method provides the step-by-step attention computation that can be
        visualized and understood for learning purposes.
        """
        # Step 1: Compute attention scores
        # scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        # Step 2: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # Step 3: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Step 4: Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Store attention weights for visualization if requested
        if store_attention:
            self.last_attention_weights = attention_weights.detach()

        # Step 5: Apply attention weights to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights if store_attention else None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights for visualization.

        Returns:
            Last attention weights tensor or None if not stored.
        """
        return self.last_attention_weights

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, temperature={self.temperature:.3f}, dropout={self.dropout.p}"
        )
