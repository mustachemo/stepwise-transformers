"""Complete transformer model implementation using PyTorch.

This module provides the full transformer architecture combining encoder and decoder
stacks with comprehensive ClearML integration and educational features.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from stepwise_transformers.layers import TransformerEncoderLayer, TransformerDecoderLayer
from stepwise_transformers.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding


class Transformer(nn.Module):
    """Complete transformer model with encoder and decoder stacks.
    
    This implementation provides both PyTorch's native Transformer and a custom
    implementation with detailed analysis capabilities for educational purposes.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        src_vocab_size: int = 32000,
        tgt_vocab_size: int = 32000,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        use_pytorch_native: bool = True,
        positional_encoding_type: str = "sinusoidal",
        feed_forward_type: str = "standard",
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        """Initialize transformer model.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_encoder_layers: Number of encoder layers.
            n_decoder_layers: Number of decoder layers.
            d_ff: Feed-forward dimension.
            src_vocab_size: Source vocabulary size.
            tgt_vocab_size: Target vocabulary size.
            max_seq_length: Maximum sequence length.
            dropout: Dropout probability.
            activation: Activation function.
            layer_norm_eps: Layer normalization epsilon.
            norm_first: Whether to apply layer norm before sublayers.
            use_pytorch_native: Whether to use PyTorch's native implementation.
            positional_encoding_type: Type of positional encoding ("sinusoidal" or "learned").
            feed_forward_type: Type of feed-forward network ("standard" or "gated").
            pad_token_id: Padding token ID.
            **kwargs: Additional arguments for feed-forward networks.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__()
        
        # Validate parameters
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_encoder_layers < 0:
            raise ValueError(f"n_encoder_layers must be non-negative, got {n_encoder_layers}")
        if n_decoder_layers < 0:
            raise ValueError(f"n_decoder_layers must be non-negative, got {n_decoder_layers}")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.use_pytorch_native = use_pytorch_native

        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        if positional_encoding_type == "sinusoidal":
            self.src_pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
                dropout=dropout,
            )
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
                dropout=dropout,
            )
        elif positional_encoding_type == "learned":
            self.src_pos_encoding = LearnedPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
                dropout=dropout,
                padding_idx=pad_token_id,
            )
            self.tgt_pos_encoding = LearnedPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
                dropout=dropout,
                padding_idx=pad_token_id,
            )
        else:
            raise ValueError(f"Unsupported positional_encoding_type: {positional_encoding_type}")

        if use_pytorch_native:
            # Use PyTorch's native transformer
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=n_heads,
                num_encoder_layers=n_encoder_layers,
                num_decoder_layers=n_decoder_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                batch_first=True,
            )
        else:
            # Custom implementation with educational features
            self._build_custom_transformer(
                d_model, n_heads, n_encoder_layers, n_decoder_layers,
                d_ff, dropout, activation, layer_norm_eps, norm_first,
                feed_forward_type, **kwargs
            )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        
        # Initialize parameters
        self._initialize_parameters()

    def _build_custom_transformer(
        self,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        norm_first: bool,
        feed_forward_type: str,
        **kwargs,
    ) -> None:
        """Build custom transformer with educational features."""
        # Encoder layers
        if n_encoder_layers > 0:
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    norm_first=norm_first,
                    use_pytorch_native=False,
                    feed_forward_type=feed_forward_type,
                    **kwargs,
                )
                for _ in range(n_encoder_layers)
            ])
            self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Decoder layers
        if n_decoder_layers > 0:
            self.decoder_layers = nn.ModuleList([
                TransformerDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    norm_first=norm_first,
                    use_pytorch_native=False,
                    feed_forward_type=feed_forward_type,
                    **kwargs,
                )
                for _ in range(n_decoder_layers)
            ])
            self.decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def _initialize_parameters(self) -> None:
        """Initialize model parameters using Xavier initialization."""
        # Initialize embeddings
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        
        # Zero out padding token embeddings
        with torch.no_grad():
            self.src_embedding.weight[self.pad_token_id].fill_(0)
            self.tgt_embedding.weight[self.pad_token_id].fill_(0)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        
        # Scale embeddings by sqrt(d_model) as in the original paper
        self.embedding_scale = math.sqrt(self.d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
        store_activations: bool = False,
    ) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            src: Source token IDs of shape (batch_size, src_seq_len).
            tgt: Target token IDs of shape (batch_size, tgt_seq_len).
            src_mask: Source attention mask.
            tgt_mask: Target attention mask (causal mask).
            memory_mask: Memory attention mask.
            src_key_padding_mask: Source key padding mask.
            tgt_key_padding_mask: Target key padding mask.
            memory_key_padding_mask: Memory key padding mask.
            store_attention: Whether to store attention weights.
            store_activations: Whether to store activations.

        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size).

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        # Validate inputs
        if src.dim() != 2:
            raise ValueError(f"Expected 2D source input (batch, seq), got {src.dim()}D")
        if tgt.dim() != 2:
            raise ValueError(f"Expected 2D target input (batch, seq), got {tgt.dim()}D")
        
        batch_size_src, src_seq_len = src.shape
        batch_size_tgt, tgt_seq_len = tgt.shape
        
        if batch_size_src != batch_size_tgt:
            raise ValueError(f"Source and target batch sizes must match: {batch_size_src} vs {batch_size_tgt}")
        
        # Embed tokens
        src_embedded = self.src_embedding(src) * self.embedding_scale
        tgt_embedded = self.tgt_embedding(tgt) * self.embedding_scale
        
        # Add positional encoding
        src_embedded = self.src_pos_encoding(src_embedded)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        
        if self.use_pytorch_native:
            # Use PyTorch's native implementation
            output = self.transformer(
                src=src_embedded,
                tgt=tgt_embedded,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            # Custom implementation
            output = self._forward_custom(
                src_embedded, tgt_embedded, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask,
                store_attention, store_activations
            )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits

    def _forward_custom(
        self,
        src_embedded: torch.Tensor,
        tgt_embedded: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
        store_attention: bool,
        store_activations: bool,
    ) -> torch.Tensor:
        """Custom forward pass with educational features."""
        # Encoder
        memory = src_embedded
        if hasattr(self, "encoder_layers"):
            for layer in self.encoder_layers:
                memory = layer(
                    src=memory,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    store_attention=store_attention,
                    store_activations=store_activations,
                )
            memory = self.encoder_norm(memory)
        
        # Decoder
        output = tgt_embedded
        if hasattr(self, "decoder_layers"):
            for layer in self.decoder_layers:
                output = layer(
                    tgt=output,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    store_attention=store_attention,
                    store_activations=store_activations,
                )
            output = self.decoder_norm(output)
        
        return output

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention.

        Args:
            size: Sequence length.

        Returns:
            Causal mask of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create padding mask from token IDs.

        Args:
            tokens: Token IDs of shape (batch_size, seq_len).

        Returns:
            Padding mask of shape (batch_size, seq_len).
        """
        return tokens == self.pad_token_id

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src: Source token IDs.
            src_mask: Source attention mask.
            src_key_padding_mask: Source key padding mask.

        Returns:
            Encoded memory.
        """
        src_embedded = self.src_embedding(src) * self.embedding_scale
        src_embedded = self.src_pos_encoding(src_embedded)
        
        if self.use_pytorch_native:
            return self.transformer.encoder(
                src=src_embedded,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            memory = src_embedded
            if hasattr(self, "encoder_layers"):
                for layer in self.encoder_layers:
                    memory = layer(
                        src=memory,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                memory = self.encoder_norm(memory)
            return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence.

        Args:
            tgt: Target token IDs.
            memory: Encoded memory from encoder.
            tgt_mask: Target attention mask.
            memory_mask: Memory attention mask.
            tgt_key_padding_mask: Target key padding mask.
            memory_key_padding_mask: Memory key padding mask.

        Returns:
            Decoded output.
        """
        tgt_embedded = self.tgt_embedding(tgt) * self.embedding_scale
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        
        if self.use_pytorch_native:
            output = self.transformer.decoder(
                tgt=tgt_embedded,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            output = tgt_embedded
            if hasattr(self, "decoder_layers"):
                for layer in self.decoder_layers:
                    output = layer(
                        tgt=output,
                        memory=memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                    )
                output = self.decoder_norm(output)
        
        return self.output_projection(output)

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        param_counts = self.count_parameters()
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_encoder_layers={self.n_encoder_layers}, n_decoder_layers={self.n_decoder_layers}, "
            f"src_vocab_size={self.src_vocab_size}, tgt_vocab_size={self.tgt_vocab_size}, "
            f"total_params={param_counts['total_parameters']:,}"
        )
