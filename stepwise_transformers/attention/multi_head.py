"""Multi-head attention implementation using PyTorch.

This module provides a wrapper around PyTorch's MultiheadAttention
with additional educational features and ClearML integration.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention with educational features and visualization support.

    This implementation leverages PyTorch's optimized MultiheadAttention
    module while adding comprehensive logging and visualization capabilities
    for learning purposes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
    ) -> None:
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.
            bias: Whether to use bias in linear projections.
            kdim: Key dimension. If None, uses d_model.
            vdim: Value dimension. If None, uses d_model.
            batch_first: If True, input shape is (batch, seq, feature).

        Raises:
            ValueError: If d_model is not divisible by n_heads or parameters are invalid.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first

        # Use PyTorch's optimized MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
        )

        # For storing attention weights for visualization
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        store_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_padding_mask: Mask for padding tokens in keys.
            attn_mask: Attention mask for preventing attention to certain positions.
            need_weights: Whether to return attention weights.
            average_attn_weights: Whether to average attention weights across heads.
            store_attention: Whether to store attention weights for visualization.

        Returns:
            Tuple of (attention_output, attention_weights).
            attention_weights is None unless need_weights=True or store_attention=True.

        Raises:
            ValueError: If tensor dimensions are incompatible.
        """
        # Validate input shapes based on batch_first setting
        if self.batch_first:
            batch_size, seq_len, feature_dim = query.shape
            if feature_dim != self.d_model:
                raise ValueError(
                    f"Query feature dimension {feature_dim} doesn't match d_model {self.d_model}"
                )
        else:
            seq_len, batch_size, feature_dim = query.shape
            if feature_dim != self.d_model:
                raise ValueError(
                    f"Query feature dimension {feature_dim} doesn't match d_model {self.d_model}"
                )

        # Determine if we need attention weights
        return_weights = need_weights or store_attention

        # Use PyTorch's optimized implementation
        output, attention_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_weights,
            average_attn_weights=average_attn_weights,
        )

        # Store attention weights for visualization if requested
        if store_attention and attention_weights is not None:
            self.last_attention_weights = attention_weights.detach()

        # Return weights only if explicitly requested
        return_weights_tensor = attention_weights if need_weights else None

        return output, return_weights_tensor

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights for visualization.

        Returns:
            Last attention weights tensor or None if not stored.
        """
        return self.last_attention_weights

    def get_head_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for individual heads (if available).

        Returns:
            Attention weights per head or None if not available.
            Shape: (batch_size, n_heads, seq_len, seq_len)
        """
        if self.last_attention_weights is None:
            return None

        # If weights are averaged, we can't recover individual head weights
        # This would require a custom implementation to store non-averaged weights
        return self.last_attention_weights

    def compute_attention_statistics(self) -> dict[str, float]:
        """Compute statistics about the last attention computation.

        Returns:
            Dictionary with attention statistics for analysis.
        """
        if self.last_attention_weights is None:
            return {}

        weights = self.last_attention_weights
        stats = {
            "attention_entropy": self._compute_attention_entropy(weights),
            "attention_max": weights.max().item(),
            "attention_min": weights.min().item(),
            "attention_std": weights.std().item(),
            "attention_sparsity": (weights < 0.01).float().mean().item(),
        }

        return stats

    def _compute_attention_entropy(self, weights: torch.Tensor) -> float:
        """Compute the entropy of attention weights.

        Higher entropy indicates more distributed attention.
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        weights_safe = weights + eps
        entropy = -(weights_safe * torch.log(weights_safe)).sum(dim=-1).mean()
        return entropy.item()

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, batch_first={self.batch_first}"
        )
