"""Sinusoidal positional encoding implementation using PyTorch.

This module provides the classic sinusoidal positional encoding from
"Attention Is All You Need" with educational features and analysis capabilities.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with educational features.

    Implements the sinusoidal positional encoding from "Attention Is All You Need"
    with additional analysis and visualization capabilities for learning purposes.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        base: float = 10000.0,
    ) -> None:
        """Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimension (must be even).
            max_seq_length: Maximum sequence length to precompute.
            dropout: Dropout probability applied after adding positional encoding.
            base: Base for the sinusoidal functions (usually 10000).

        Raises:
            ValueError: If d_model is not even or other parameters are invalid.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal encoding, got {d_model}")
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.base = base
        self.dropout = nn.Dropout(dropout)

        # Precompute positional encodings
        pe = self._create_positional_encoding()

        # Register as buffer so it moves with the model but isn't a parameter
        self.register_buffer("pe", pe, persistent=False)

    def _create_positional_encoding(self) -> torch.Tensor:
        """Create the sinusoidal positional encoding matrix.

        Returns:
            Positional encoding matrix of shape (max_seq_length, d_model).
        """
        pe = torch.zeros(self.max_seq_length, self.d_model)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)

        # Create the div_term for the sinusoidal functions
        # div_term = exp(-log(base) * 2i / d_model) for i in [0, 1, ..., d_model//2-1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float)
            * -(math.log(self.base) / self.d_model)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            positions: Optional position indices. If None, uses sequential positions.

        Returns:
            Input tensor with positional encoding added.

        Raises:
            ValueError: If sequence length exceeds max_seq_length or dimensions mismatch.
        """
        batch_size, seq_len, feature_dim = x.shape

        if feature_dim != self.d_model:
            raise ValueError(
                f"Input feature dimension {feature_dim} doesn't match d_model {self.d_model}"
            )
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_length {self.max_seq_length}"
            )

        if positions is None:
            # Use sequential positions
            pe = self.pe[:seq_len].unsqueeze(0)  # Shape: (1, seq_len, d_model)
        else:
            # Use provided positions
            if positions.max().item() >= self.max_seq_length:
                raise ValueError(
                    f"Position index {positions.max().item()} exceeds max_seq_length {self.max_seq_length}"
                )
            pe = self.pe[positions]  # Shape: (batch_size, seq_len, d_model)

        # Add positional encoding to input
        x = x + pe

        return self.dropout(x)

    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for a specific sequence length.

        Args:
            seq_len: Sequence length.

        Returns:
            Positional encoding tensor of shape (seq_len, d_model).

        Raises:
            ValueError: If seq_len exceeds max_seq_length.
        """
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_length {self.max_seq_length}"
            )

        return self.pe[:seq_len].clone()

    def visualize_encoding(self, seq_len: int = 100) -> dict[str, torch.Tensor]:
        """Get data for visualizing the positional encoding patterns.

        Args:
            seq_len: Sequence length to visualize.

        Returns:
            Dictionary with visualization data including encoding matrix and frequencies.
        """
        if seq_len > self.max_seq_length:
            seq_len = self.max_seq_length

        pe = self.pe[:seq_len]

        # Compute frequencies for each dimension
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float)
            * -(math.log(self.base) / self.d_model)
        )

        return {
            "encoding_matrix": pe,
            "frequencies": frequencies,
            "positions": torch.arange(seq_len),
            "d_model": self.d_model,
        }

    def compute_similarity_matrix(self, seq_len: int = 50) -> torch.Tensor:
        """Compute cosine similarity between positional encodings.

        This helps understand how similar positions are encoded.

        Args:
            seq_len: Sequence length to analyze.

        Returns:
            Similarity matrix of shape (seq_len, seq_len).
        """
        if seq_len > self.max_seq_length:
            seq_len = self.max_seq_length

        pe = self.pe[:seq_len]  # Shape: (seq_len, d_model)

        # Compute cosine similarity
        pe_norm = torch.nn.functional.normalize(pe, p=2, dim=1)
        similarity = torch.matmul(pe_norm, pe_norm.transpose(0, 1))

        return similarity

    def extend_max_length(self, new_max_length: int) -> None:
        """Extend the maximum sequence length by recomputing encodings.

        Args:
            new_max_length: New maximum sequence length.

        Raises:
            ValueError: If new_max_length is not greater than current max_length.
        """
        if new_max_length <= self.max_seq_length:
            raise ValueError(
                f"New max length {new_max_length} must be greater than current {self.max_seq_length}"
            )

        self.max_seq_length = new_max_length
        pe = self._create_positional_encoding()
        self.register_buffer("pe", pe, persistent=False)

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, max_seq_length={self.max_seq_length}, "
            f"base={self.base}, dropout={self.dropout.p}"
        )
