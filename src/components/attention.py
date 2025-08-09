"""
Standard attention mechanism implementation for transformers.

This module implements the scaled dot-product attention mechanism as described
in "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention as: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    where Q, K, V are query, key, and value matrices respectively.
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        """
        Initialize the attention mechanism.
        
        Args:
            dropout_rate: Dropout rate for attention weights.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_k)
            key: Key tensor of shape (batch_size, seq_len, d_k)
            value: Value tensor of shape (batch_size, seq_len, d_v)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                 Values should be 0 for valid positions and -inf for masked positions.
        
        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, seq_len, d_v)
                - Attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_k = query.size()
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Runs multiple attention heads in parallel and concatenates their outputs.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout_rate: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
            
        Raises:
            ValueError: If d_model is not divisible by n_heads
        """
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout_rate)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
        
        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multi-head if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, seq_len, seq_len)
        
        # Apply attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        return output, attention_weights


def create_padding_mask(sequences: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create padding mask for attention mechanism.
    
    Args:
        sequences: Input sequences of shape (batch_size, seq_len)
        pad_token: Token ID used for padding
    
    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
        where masked positions are False and valid positions are True
    """
    batch_size, seq_len = sequences.size()
    
    # Create mask where padding tokens are False
    padding_mask = (sequences != pad_token).unsqueeze(1).unsqueeze(2)
    padding_mask = padding_mask.expand(batch_size, seq_len, seq_len)
    
    return padding_mask


def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create look-ahead mask for decoder self-attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create the mask on
    
    Returns:
        Look-ahead mask of shape (seq_len, seq_len)
        where future positions are False and current/past positions are True
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0
