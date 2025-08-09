"""Transformer encoder layer implementation using PyTorch.

This module provides a wrapper around PyTorch's TransformerEncoderLayer
with additional educational features and analysis capabilities.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from stepwise_transformers.attention import MultiHeadAttention
from stepwise_transformers.feed_forward import PositionWiseFeedForward, GatedFeedForward


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with educational features.

    This implementation provides both PyTorch's optimized TransformerEncoderLayer
    and a custom implementation with detailed analysis capabilities for learning.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        use_pytorch_native: bool = True,
        feed_forward_type: str = "standard",
        **kwargs,
    ) -> None:
        """Initialize transformer encoder layer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            activation: Activation function for feed-forward network.
            layer_norm_eps: Layer normalization epsilon.
            norm_first: Whether to apply layer norm before or after sublayers.
            use_pytorch_native: Whether to use PyTorch's native implementation.
            feed_forward_type: Type of feed-forward network ("standard" or "gated").
            **kwargs: Additional arguments for feed-forward networks.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.norm_first = norm_first
        self.use_pytorch_native = use_pytorch_native

        if use_pytorch_native:
            # Use PyTorch's optimized implementation
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                batch_first=True,
            )
        else:
            # Custom implementation with educational features
            self._build_custom_layer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                activation,
                layer_norm_eps,
                norm_first,
                feed_forward_type,
                **kwargs,
            )

    def _build_custom_layer(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        norm_first: bool,
        feed_forward_type: str,
        **kwargs,
    ) -> None:
        """Build custom encoder layer components."""
        # Multi-head attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        if feed_forward_type == "standard":
            self.feed_forward = PositionWiseFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
        elif feed_forward_type == "gated":
            glu_variant = kwargs.get("glu_variant", "swiglu")
            self.feed_forward = GatedFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                glu_variant=glu_variant,
            )
        else:
            raise ValueError(f"Unsupported feed_forward_type: {feed_forward_type}")

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
        store_activations: bool = False,
    ) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            src: Source tensor of shape (batch_size, seq_len, d_model).
            src_mask: Source attention mask.
            src_key_padding_mask: Source key padding mask.
            store_attention: Whether to store attention weights for visualization.
            store_activations: Whether to store activations for analysis.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        if src.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {src.dim()}D")

        batch_size, seq_len, feature_dim = src.shape
        if feature_dim != self.d_model:
            raise ValueError(
                f"Input feature dimension {feature_dim} doesn't match d_model {self.d_model}"
            )

        if self.use_pytorch_native:
            return self.encoder_layer(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            return self._forward_custom(
                src, src_mask, src_key_padding_mask, store_attention, store_activations
            )

    def _forward_custom(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
        store_attention: bool,
        store_activations: bool,
    ) -> torch.Tensor:
        """Custom forward pass with educational features."""
        if self.norm_first:
            # Pre-norm architecture
            # Self-attention block
            norm_src = self.norm1(src)
            attn_output, _ = self.self_attention(
                query=norm_src,
                key=norm_src,
                value=norm_src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                store_attention=store_attention,
            )
            src = src + self.dropout(attn_output)

            # Feed-forward block
            norm_src = self.norm2(src)
            ff_output = self.feed_forward(norm_src, store_activations=store_activations)
            src = src + self.dropout(ff_output)
        else:
            # Post-norm architecture
            # Self-attention block
            attn_output, _ = self.self_attention(
                query=src,
                key=src,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                store_attention=store_attention,
            )
            src = self.norm1(src + self.dropout(attn_output))

            # Feed-forward block
            ff_output = self.feed_forward(src, store_activations=store_activations)
            src = self.norm2(src + self.dropout(ff_output))

        return src

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from the self-attention layer.

        Returns:
            Attention weights or None if not available.
        """
        if hasattr(self, "self_attention"):
            return self.self_attention.get_attention_weights()
        return None

    def get_feed_forward_activations(self) -> Optional[torch.Tensor]:
        """Get feed-forward activations for analysis.

        Returns:
            Feed-forward activations or None if not available.
        """
        if hasattr(self, "feed_forward"):
            if hasattr(self.feed_forward, "get_hidden_activations"):
                return self.feed_forward.get_hidden_activations()
            elif hasattr(self.feed_forward, "get_activations"):
                return self.feed_forward.get_activations()
        return None

    def compute_layer_statistics(self) -> dict[str, float]:
        """Compute statistics about the layer's components.

        Returns:
            Dictionary with layer statistics for analysis.
        """
        stats = {}

        # Attention statistics
        if hasattr(self, "self_attention"):
            attention_stats = self.self_attention.compute_attention_statistics()
            stats.update({f"attention_{k}": v for k, v in attention_stats.items()})

        # Feed-forward statistics
        if hasattr(self, "feed_forward"):
            if hasattr(self.feed_forward, "compute_activation_statistics"):
                ff_stats = self.feed_forward.compute_activation_statistics()
                stats.update({f"feed_forward_{k}": v for k, v in ff_stats.items()})

        # Layer norm statistics
        if hasattr(self, "norm1"):
            stats["norm1_weight_mean"] = self.norm1.weight.mean().item()
            stats["norm1_weight_std"] = self.norm1.weight.std().item()
            stats["norm1_bias_mean"] = self.norm1.bias.mean().item()
            stats["norm1_bias_std"] = self.norm1.bias.std().item()

        if hasattr(self, "norm2"):
            stats["norm2_weight_mean"] = self.norm2.weight.mean().item()
            stats["norm2_weight_std"] = self.norm2.weight.std().item()
            stats["norm2_bias_mean"] = self.norm2.bias.mean().item()
            stats["norm2_bias_std"] = self.norm2.bias.std().item()

        return stats

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}, "
            f"norm_first={self.norm_first}, use_pytorch_native={self.use_pytorch_native}"
        )
