"""
Feed-forward network implementation for transformers.

This module implements the position-wise feed-forward network used in
transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Applies two linear transformations with a ReLU activation in between:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Feed-forward dimension (hidden layer size)
            dropout_rate: Dropout rate applied after first linear layer
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is equivalent to Swish
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
            
        return activations[activation]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation + activation + dropout
        hidden = self.dropout(self.activation(self.linear1(x)))
        
        # Second linear transformation
        output = self.linear2(hidden)
        
        return output


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) based feed-forward network.
    
    Uses gated linear units as described in "Language Modeling with Gated 
    Convolutional Networks" (Dauphin et al., 2017).
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize the GLU feed-forward network.
        
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Feed-forward dimension (hidden layer size)
            dropout_rate: Dropout rate applied after gating
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Linear layers for gate and value
        self.gate_linear = nn.Linear(d_model, d_ff)
        self.value_linear = nn.Linear(d_model, d_ff)
        
        # Output projection
        self.output_linear = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GLU feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Compute gate and value
        gate = torch.sigmoid(self.gate_linear(x))
        value = self.value_linear(x)
        
        # Apply gating
        gated = gate * value
        
        # Apply dropout
        gated = self.dropout(gated)
        
        # Output projection
        output = self.output_linear(gated)
        
        return output
