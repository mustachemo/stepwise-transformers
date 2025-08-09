"""
Complete transformer model implementation.

This module implements the full transformer architecture with encoder and
decoder stacks as described in "Attention Is All You Need".
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.attention import create_padding_mask, create_look_ahead_mask
from ..components.positional_encoding import PositionalEncoding
from .transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack.

    Consists of multiple encoder layers stacked together.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 5000,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize transformer encoder.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_len, dropout_rate
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate, activation)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of transformer encoder.

        Args:
            src: Source tokens of shape (batch_size, src_seq_len)
            src_mask: Optional source mask

        Returns:
            Tuple containing:
                - Encoder output of shape (batch_size, src_seq_len, d_model)
                - Dictionary of attention weights for each layer
        """
        # Token embeddings
        x = self.embedding(src)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Store attention weights
        attention_weights = {}

        # Pass through encoder layers
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, src_mask)
            attention_weights[f"layer_{i}"] = attn_weights

        # Final layer normalization
        output = self.norm(x)

        return output, attention_weights


class TransformerDecoder(nn.Module):
    """
    Transformer decoder stack.

    Consists of multiple decoder layers stacked together.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 5000,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_len, dropout_rate
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout_rate, activation)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of transformer decoder.

        Args:
            tgt: Target tokens of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Optional target mask
            src_mask: Optional source mask

        Returns:
            Tuple containing:
                - Output logits of shape (batch_size, tgt_seq_len, vocab_size)
                - Dictionary of attention weights for each layer
        """
        # Token embeddings
        x = self.embedding(tgt)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Store attention weights
        attention_weights = {"self_attention": {}, "cross_attention": {}}

        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, tgt_mask, src_mask
            )
            attention_weights["self_attention"][f"layer_{i}"] = self_attn_weights
            attention_weights["cross_attention"][f"layer_{i}"] = cross_attn_weights

        # Final layer normalization
        x = self.norm(x)

        # Output projection
        output = self.output_projection(x)

        return output, attention_weights


class Transformer(nn.Module):
    """
    Complete transformer model.

    Combines encoder and decoder stacks for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        pad_token: int = 0,
    ):
        """
        Initialize transformer model.

        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder/decoder layers
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            activation: Activation function
            pad_token: Padding token ID
        """
        super().__init__()

        self.d_model = d_model
        self.pad_token = pad_token

        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
            dropout_rate,
            activation,
        )

        # Decoder
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
            dropout_rate,
            activation,
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_masks(
        self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create attention masks for source and target sequences.

        Args:
            src: Source tokens of shape (batch_size, src_seq_len)
            tgt: Optional target tokens of shape (batch_size, tgt_seq_len)

        Returns:
            Tuple containing:
                - Source padding mask
                - Target mask (if target is provided)
        """
        # Source padding mask
        src_mask = create_padding_mask(src, self.pad_token)

        if tgt is not None:
            # Target padding mask
            tgt_padding_mask = create_padding_mask(tgt, self.pad_token)

            # Target look-ahead mask
            tgt_seq_len = tgt.size(1)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq_len, tgt.device)

            # Combine masks
            tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)

            return src_mask, tgt_mask

        return src_mask, None

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass of transformer model.

        Args:
            src: Source tokens of shape (batch_size, src_seq_len)
            tgt: Optional target tokens of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Tuple containing:
                - Encoder output of shape (batch_size, src_seq_len, d_model)
                - Decoder output (if target provided) of shape (batch_size, tgt_seq_len, vocab_size)
                - Dictionary of attention weights
        """
        # Create masks if not provided
        if src_mask is None:
            if tgt is not None:
                src_mask, tgt_mask = self.create_masks(src, tgt)
            else:
                src_mask, _ = self.create_masks(src)

        # Encoder forward pass
        encoder_output, encoder_attention = self.encoder(src, src_mask)

        # Decoder forward pass (if target provided)
        decoder_output = None
        decoder_attention = {}

        if tgt is not None:
            decoder_output, decoder_attention = self.decoder(
                tgt, encoder_output, tgt_mask, src_mask
            )

        # Combine attention weights
        attention_weights = {"encoder": encoder_attention, "decoder": decoder_attention}

        return encoder_output, decoder_output, attention_weights

    def generate(
        self,
        src: torch.Tensor,
        max_length: int,
        start_token: int,
        end_token: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequence using the transformer model.

        Args:
            src: Source tokens of shape (batch_size, src_seq_len)
            max_length: Maximum length of generated sequence
            start_token: Start token ID
            end_token: End token ID
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated sequence of shape (batch_size, generated_seq_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        src_mask, _ = self.create_masks(src)
        encoder_output, _ = self.encoder(src, src_mask)

        # Initialize target with start token
        tgt = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_length - 1):
                # Create target mask
                _, tgt_mask = self.create_masks(src, tgt)

                # Decoder forward pass
                decoder_output, _ = self.decoder(
                    tgt, encoder_output, tgt_mask, src_mask
                )

                # Get next token logits
                next_token_logits = decoder_output[:, -1, :] / temperature

                # Apply top-k sampling if specified
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits.scatter_(-1, indices, values)

                # Sample next token
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)

                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)

                # Check for end token
                if (next_token == end_token).all():
                    break

        return tgt
