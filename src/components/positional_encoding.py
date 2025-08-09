"""
Positional encoding implementation for transformers.

This module implements sinusoidal positional encoding as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.
    """

    def __init__(
        self, d_model: int, max_seq_len: int = 5000, dropout_rate: float = 0.1
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension (embedding size)
            max_seq_len: Maximum sequence length supported
            dropout_rate: Dropout rate applied after adding positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Create division term for frequency calculation
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added, same shape as input
        """
        # Add positional encoding (scaled by sqrt(d_model))
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for transformer models.

    Uses learned embeddings for position information instead of
    fixed sinusoidal patterns.
    """

    def __init__(
        self, d_model: int, max_seq_len: int = 5000, dropout_rate: float = 0.1
    ):
        """
        Initialize learned positional encoding.

        Args:
            d_model: Model dimension (embedding size)
            max_seq_len: Maximum sequence length supported
            dropout_rate: Dropout rate applied after adding positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)

        # Learned position embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added, same shape as input
        """
        batch_size, seq_len, d_model = x.size()

        # Create position indices
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        # Get position embeddings
        position_encodings = self.position_embeddings(positions)

        # Add positional encoding (scaled by sqrt(d_model))
        x = x * math.sqrt(self.d_model)
        x = x + position_encodings

        return self.dropout(x)
