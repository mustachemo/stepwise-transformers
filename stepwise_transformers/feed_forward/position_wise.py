"""Position-wise feed-forward network implementation using PyTorch.

This module implements the position-wise feed-forward network used in transformers,
leveraging PyTorch's optimized linear layers with educational features.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network with educational features.
    
    Implements the standard transformer feed-forward network:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    With support for different activation functions and analysis capabilities.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        """Initialize position-wise feed-forward network.

        Args:
            d_model: Model dimension (input and output dimension).
            d_ff: Feed-forward dimension (hidden layer dimension).
            dropout: Dropout probability applied to hidden layer.
            activation: Activation function ("relu", "gelu", "swish", "mish").
            bias: Whether to use bias in linear layers.

        Raises:
            ValueError: If parameters are invalid or activation is not supported.
        """
        super().__init__()
        
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
            
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation
        
        # Define linear layers
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Get activation function
        self.activation = self._get_activation_function(activation)
        
        # For storing intermediate activations for analysis
        self.last_hidden_activations: Optional[torch.Tensor] = None

    def _get_activation_function(self, activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get activation function by name.

        Args:
            activation: Name of activation function.

        Returns:
            Activation function.

        Raises:
            ValueError: If activation is not supported.
        """
        activation_functions = {
            "relu": F.relu,
            "gelu": F.gelu,
            "swish": F.silu,  # SiLU is the same as Swish
            "mish": F.mish,
        }
        
        if activation not in activation_functions:
            supported = ", ".join(activation_functions.keys())
            raise ValueError(f"Unsupported activation '{activation}'. Supported: {supported}")
        
        return activation_functions[activation]

    def forward(
        self, 
        x: torch.Tensor, 
        store_activations: bool = False
    ) -> torch.Tensor:
        """Apply position-wise feed-forward transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            store_activations: Whether to store hidden activations for analysis.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")
        
        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(f"Input feature dimension {feature_dim} doesn't match d_model {self.d_model}")
        
        # First linear transformation
        hidden = self.linear1(x)  # Shape: (batch_size, seq_len, d_ff)
        
        # Apply activation function
        hidden = self.activation(hidden)
        
        # Store activations for analysis if requested
        if store_activations:
            self.last_hidden_activations = hidden.detach()
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Second linear transformation
        output = self.linear2(hidden)  # Shape: (batch_size, seq_len, d_model)
        
        return output

    def get_hidden_activations(self) -> Optional[torch.Tensor]:
        """Get the last computed hidden activations for analysis.

        Returns:
            Last hidden activations tensor or None if not stored.
        """
        return self.last_hidden_activations

    def compute_activation_statistics(self) -> dict[str, float]:
        """Compute statistics about the last hidden activations.

        Returns:
            Dictionary with activation statistics for analysis.
        """
        if self.last_hidden_activations is None:
            return {}
        
        activations = self.last_hidden_activations
        stats = {
            "activation_mean": activations.mean().item(),
            "activation_std": activations.std().item(),
            "activation_max": activations.max().item(),
            "activation_min": activations.min().item(),
            "dead_neurons_ratio": (activations == 0).float().mean().item(),
            "positive_activations_ratio": (activations > 0).float().mean().item(),
        }
        
        # Activation-specific statistics
        if self.activation_name == "relu":
            stats["relu_saturation"] = (activations == 0).float().mean().item()
        elif self.activation_name == "gelu":
            # GELU rarely has exact zeros, so use small threshold
            stats["near_zero_activations"] = (activations.abs() < 1e-6).float().mean().item()
        
        return stats

    def compute_weight_statistics(self) -> dict[str, float]:
        """Compute statistics about the linear layer weights.

        Returns:
            Dictionary with weight statistics for analysis.
        """
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        
        stats = {
            "linear1_weight_norm": w1.norm().item(),
            "linear1_weight_mean": w1.mean().item(),
            "linear1_weight_std": w1.std().item(),
            "linear2_weight_norm": w2.norm().item(),
            "linear2_weight_mean": w2.mean().item(),
            "linear2_weight_std": w2.std().item(),
        }
        
        if self.linear1.bias is not None:
            stats["linear1_bias_norm"] = self.linear1.bias.norm().item()
            stats["linear2_bias_norm"] = self.linear2.bias.norm().item()
        
        return stats

    def reset_parameters(self) -> None:
        """Reset all parameters to their initial values using Xavier initialization."""
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        
        # Zero initialization for biases
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"activation={self.activation_name}, dropout={self.dropout.p}"
        )
