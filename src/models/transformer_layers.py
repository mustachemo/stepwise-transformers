"""
Transformer encoder and decoder layer implementations.

This module implements the encoder and decoder layers that form the building
blocks of the transformer architecture.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..components.attention import MultiHeadAttention
from ..components.feed_forward import PositionWiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Consists of multi-head self-attention followed by position-wise
    feed-forward network, with residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize transformer encoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout_rate: Dropout rate
            activation: Activation function for feed-forward network
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_rate)

        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, dropout_rate, activation
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights
        """
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        output = self.norm2(x + ff_output)

        return output, attn_weights


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Consists of masked multi-head self-attention, encoder-decoder attention,
    and position-wise feed-forward network, with residual connections and
    layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize transformer decoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout_rate: Dropout rate
            activation: Activation function for feed-forward network
        """
        super().__init__()

        # Multi-head self-attention (masked)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_rate)

        # Multi-head encoder-decoder attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout_rate)

        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, dropout_rate, activation
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, enc_seq_len, d_model)
            self_attn_mask: Optional self-attention mask
            cross_attn_mask: Optional cross-attention mask

        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Self-attention weights
                - Cross-attention weights
        """
        # Masked self-attention with residual connection and layer norm
        self_attn_output, self_attn_weights = self.self_attention(
            x, x, x, self_attn_mask
        )
        self_attn_output = self.dropout1(self_attn_output)
        x = self.norm1(x + self_attn_output)

        # Encoder-decoder attention with residual connection and layer norm
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, cross_attn_mask
        )
        cross_attn_output = self.dropout2(cross_attn_output)
        x = self.norm2(x + cross_attn_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        output = self.norm3(x + ff_output)

        return output, self_attn_weights, cross_attn_weights
