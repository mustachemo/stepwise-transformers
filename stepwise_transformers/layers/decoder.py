"""Transformer decoder layer implementation using PyTorch.

This module provides a wrapper around PyTorch's TransformerDecoderLayer
with additional educational features and analysis capabilities.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from stepwise_transformers.attention import MultiHeadAttention
from stepwise_transformers.feed_forward import PositionWiseFeedForward, GatedFeedForward


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with educational features.

    This implementation provides both PyTorch's optimized TransformerDecoderLayer
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
        """Initialize transformer decoder layer.

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
            self.decoder_layer = nn.TransformerDecoderLayer(
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
        """Build custom decoder layer components."""
        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
        store_activations: bool = False,
    ) -> torch.Tensor:
        """Forward pass through decoder layer.

        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len, d_model).
            memory: Memory tensor from encoder of shape (batch_size, src_seq_len, d_model).
            tgt_mask: Target attention mask for causal masking.
            memory_mask: Memory attention mask.
            tgt_key_padding_mask: Target key padding mask.
            memory_key_padding_mask: Memory key padding mask.
            store_attention: Whether to store attention weights for visualization.
            store_activations: Whether to store activations for analysis.

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model).

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        if tgt.dim() != 3:
            raise ValueError(f"Expected 3D target input (batch, seq, features), got {tgt.dim()}D")
        if memory.dim() != 3:
            raise ValueError(
                f"Expected 3D memory input (batch, seq, features), got {memory.dim()}D"
            )

        batch_size_tgt, tgt_seq_len, tgt_feature_dim = tgt.shape
        batch_size_mem, mem_seq_len, mem_feature_dim = memory.shape

        if tgt_feature_dim != self.d_model:
            raise ValueError(
                f"Target feature dimension {tgt_feature_dim} doesn't match d_model {self.d_model}"
            )
        if mem_feature_dim != self.d_model:
            raise ValueError(
                f"Memory feature dimension {mem_feature_dim} doesn't match d_model {self.d_model}"
            )
        if batch_size_tgt != batch_size_mem:
            raise ValueError(
                f"Target and memory batch sizes must match: {batch_size_tgt} vs {batch_size_mem}"
            )

        if self.use_pytorch_native:
            return self.decoder_layer(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            return self._forward_custom(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                store_attention,
                store_activations,
            )

    def _forward_custom(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
        store_attention: bool,
        store_activations: bool,
    ) -> torch.Tensor:
        """Custom forward pass with educational features."""
        if self.norm_first:
            # Pre-norm architecture
            # Self-attention block (masked)
            norm_tgt = self.norm1(tgt)
            self_attn_output, _ = self.self_attention(
                query=norm_tgt,
                key=norm_tgt,
                value=norm_tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                store_attention=store_attention,
            )
            tgt = tgt + self.dropout(self_attn_output)

            # Cross-attention block
            norm_tgt = self.norm2(tgt)
            cross_attn_output, _ = self.cross_attention(
                query=norm_tgt,
                key=memory,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                store_attention=store_attention,
            )
            tgt = tgt + self.dropout(cross_attn_output)

            # Feed-forward block
            norm_tgt = self.norm3(tgt)
            ff_output = self.feed_forward(norm_tgt, store_activations=store_activations)
            tgt = tgt + self.dropout(ff_output)
        else:
            # Post-norm architecture
            # Self-attention block (masked)
            self_attn_output, _ = self.self_attention(
                query=tgt,
                key=tgt,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                store_attention=store_attention,
            )
            tgt = self.norm1(tgt + self.dropout(self_attn_output))

            # Cross-attention block
            cross_attn_output, _ = self.cross_attention(
                query=tgt,
                key=memory,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                store_attention=store_attention,
            )
            tgt = self.norm2(tgt + self.dropout(cross_attn_output))

            # Feed-forward block
            ff_output = self.feed_forward(tgt, store_activations=store_activations)
            tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt

    def get_self_attention_weights(self) -> Optional[torch.Tensor]:
        """Get self-attention weights for visualization.

        Returns:
            Self-attention weights or None if not available.
        """
        if hasattr(self, "self_attention"):
            return self.self_attention.get_attention_weights()
        return None

    def get_cross_attention_weights(self) -> Optional[torch.Tensor]:
        """Get cross-attention weights for visualization.

        Returns:
            Cross-attention weights or None if not available.
        """
        if hasattr(self, "cross_attention"):
            return self.cross_attention.get_attention_weights()
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

        # Self-attention statistics
        if hasattr(self, "self_attention"):
            self_attn_stats = self.self_attention.compute_attention_statistics()
            stats.update({f"self_attention_{k}": v for k, v in self_attn_stats.items()})

        # Cross-attention statistics
        if hasattr(self, "cross_attention"):
            cross_attn_stats = self.cross_attention.compute_attention_statistics()
            stats.update({f"cross_attention_{k}": v for k, v in cross_attn_stats.items()})

        # Feed-forward statistics
        if hasattr(self, "feed_forward"):
            if hasattr(self.feed_forward, "compute_activation_statistics"):
                ff_stats = self.feed_forward.compute_activation_statistics()
                stats.update({f"feed_forward_{k}": v for k, v in ff_stats.items()})

        # Layer norm statistics
        for i, norm in enumerate(
            [
                getattr(self, "norm1", None),
                getattr(self, "norm2", None),
                getattr(self, "norm3", None),
            ],
            1,
        ):
            if norm is not None:
                stats[f"norm{i}_weight_mean"] = norm.weight.mean().item()
                stats[f"norm{i}_weight_std"] = norm.weight.std().item()
                stats[f"norm{i}_bias_mean"] = norm.bias.mean().item()
                stats[f"norm{i}_bias_std"] = norm.bias.std().item()

        return stats

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}, "
            f"norm_first={self.norm_first}, use_pytorch_native={self.use_pytorch_native}"
        )
