"""Learned positional encoding implementation using PyTorch.

This module provides learned positional embeddings as an alternative to
sinusoidal encoding, with educational features and analysis capabilities.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding with educational features.
    
    Implements learnable positional embeddings that can be trained alongside
    the model, with additional analysis and visualization capabilities.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        init_std: float = 0.02,
    ) -> None:
        """Initialize learned positional encoding.

        Args:
            d_model: Model dimension.
            max_seq_length: Maximum sequence length.
            dropout: Dropout probability applied after adding positional encoding.
            padding_idx: If specified, the entries at this index are masked out with zeros.
            init_std: Standard deviation for normal initialization of embeddings.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__()
        
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        if padding_idx is not None and not (0 <= padding_idx < max_seq_length):
            raise ValueError(f"padding_idx must be in [0, {max_seq_length}), got {padding_idx}")
            
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.dropout = nn.Dropout(dropout)
        
        # Create learnable position embeddings
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_seq_length,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        
        # Initialize embeddings
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Initialize position embeddings with normal distribution."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=self.init_std)
        
        # Zero out padding position if specified
        if self.padding_idx is not None:
            with torch.no_grad():
                self.position_embeddings.weight[self.padding_idx].fill_(0)

    def forward(
        self, 
        x: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            positions: Optional position indices of shape (batch_size, seq_len).
                      If None, uses sequential positions starting from 0.

        Returns:
            Input tensor with positional encoding added.

        Raises:
            ValueError: If sequence length exceeds max_seq_length or dimensions mismatch.
        """
        batch_size, seq_len, feature_dim = x.shape
        
        if feature_dim != self.d_model:
            raise ValueError(f"Input feature dimension {feature_dim} doesn't match d_model {self.d_model}")
        if seq_len > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_length {self.max_seq_length}")
        
        if positions is None:
            # Create sequential positions
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        else:
            # Validate provided positions
            if positions.shape != (batch_size, seq_len):
                raise ValueError(f"Positions shape {positions.shape} doesn't match input shape ({batch_size}, {seq_len})")
            if positions.max().item() >= self.max_seq_length:
                raise ValueError(f"Position index {positions.max().item()} exceeds max_seq_length {self.max_seq_length}")
        
        # Get position embeddings
        pe = self.position_embeddings(positions)  # Shape: (batch_size, seq_len, d_model)
        
        # Add positional encoding to input
        x = x + pe
        
        return self.dropout(x)

    def get_position_embedding(self, position: int) -> torch.Tensor:
        """Get embedding for a specific position.

        Args:
            position: Position index.

        Returns:
            Position embedding tensor of shape (d_model,).

        Raises:
            ValueError: If position exceeds max_seq_length.
        """
        if position >= self.max_seq_length:
            raise ValueError(f"Position {position} exceeds max_seq_length {self.max_seq_length}")
        
        return self.position_embeddings.weight[position].clone()

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all position embeddings.

        Returns:
            All position embeddings of shape (max_seq_length, d_model).
        """
        return self.position_embeddings.weight.clone()

    def compute_similarity_matrix(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """Compute cosine similarity between position embeddings.
        
        This helps understand how the model has learned to represent positions.

        Args:
            seq_len: Sequence length to analyze. If None, uses max_seq_length.

        Returns:
            Similarity matrix of shape (seq_len, seq_len).
        """
        if seq_len is None:
            seq_len = self.max_seq_length
        elif seq_len > self.max_seq_length:
            seq_len = self.max_seq_length
            
        embeddings = self.position_embeddings.weight[:seq_len]  # Shape: (seq_len, d_model)
        
        # Compute cosine similarity
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1))
        
        return similarity

    def compute_embedding_statistics(self) -> dict[str, float]:
        """Compute statistics about the learned embeddings.

        Returns:
            Dictionary with embedding statistics for analysis.
        """
        embeddings = self.position_embeddings.weight
        
        stats = {
            "embedding_norm_mean": embeddings.norm(dim=1).mean().item(),
            "embedding_norm_std": embeddings.norm(dim=1).std().item(),
            "embedding_mean": embeddings.mean().item(),
            "embedding_std": embeddings.std().item(),
            "embedding_max": embeddings.max().item(),
            "embedding_min": embeddings.min().item(),
        }
        
        # Compute pairwise distances between consecutive positions
        if self.max_seq_length > 1:
            consecutive_distances = torch.norm(
                embeddings[1:] - embeddings[:-1], dim=1
            )
            stats["consecutive_distance_mean"] = consecutive_distances.mean().item()
            stats["consecutive_distance_std"] = consecutive_distances.std().item()
        
        return stats

    def extend_max_length(self, new_max_length: int) -> None:
        """Extend the maximum sequence length by adding new embeddings.

        Args:
            new_max_length: New maximum sequence length.

        Raises:
            ValueError: If new_max_length is not greater than current max_length.
        """
        if new_max_length <= self.max_seq_length:
            raise ValueError(f"New max length {new_max_length} must be greater than current {self.max_seq_length}")
        
        # Create new embedding layer with extended length
        new_embeddings = nn.Embedding(
            num_embeddings=new_max_length,
            embedding_dim=self.d_model,
            padding_idx=self.padding_idx,
        )
        
        # Copy existing embeddings
        with torch.no_grad():
            new_embeddings.weight[:self.max_seq_length] = self.position_embeddings.weight
            
            # Initialize new positions with normal distribution
            nn.init.normal_(
                new_embeddings.weight[self.max_seq_length:], 
                mean=0.0, 
                std=self.init_std
            )
        
        # Replace the embedding layer
        self.position_embeddings = new_embeddings
        self.max_seq_length = new_max_length

    def freeze_embeddings(self) -> None:
        """Freeze position embeddings to prevent further training."""
        self.position_embeddings.weight.requires_grad = False

    def unfreeze_embeddings(self) -> None:
        """Unfreeze position embeddings to allow training."""
        self.position_embeddings.weight.requires_grad = True

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, max_seq_length={self.max_seq_length}, "
            f"padding_idx={self.padding_idx}, dropout={self.dropout.p}"
        )
