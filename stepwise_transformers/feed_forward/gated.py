"""Gated feed-forward network implementation using PyTorch.

This module implements gated linear unit (GLU) variants used in modern transformers,
providing more expressive feed-forward networks with gating mechanisms.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeedForward(nn.Module):
    """Gated feed-forward network with various GLU variants.

    Implements different variants of Gated Linear Units (GLUs):
    - GLU: GLU(x) = (xW + b) ⊙ σ(xV + c)
    - GEGLU: GEGLU(x) = GELU(xW + b) ⊙ (xV + c)
    - SwiGLU: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)

    Where ⊙ denotes element-wise multiplication and σ is sigmoid.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        glu_variant: str = "swiglu",
        bias: bool = True,
        gate_bias: bool = True,
    ) -> None:
        """Initialize gated feed-forward network.

        Args:
            d_model: Model dimension (input and output dimension).
            d_ff: Feed-forward dimension (hidden layer dimension).
            dropout: Dropout probability applied to gated activations.
            glu_variant: GLU variant ("glu", "geglu", "swiglu").
            bias: Whether to use bias in main linear layers.
            gate_bias: Whether to use bias in gate linear layer.

        Raises:
            ValueError: If parameters are invalid or GLU variant is not supported.
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
        self.glu_variant = glu_variant

        # Define linear layers
        self.linear_main = nn.Linear(d_model, d_ff, bias=bias)
        self.linear_gate = nn.Linear(d_model, d_ff, bias=gate_bias)
        self.linear_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Get activation and gate functions
        self.activation_fn, self.gate_fn = self._get_activation_functions(glu_variant)

        # For storing intermediate activations for analysis
        self.last_main_activations: Optional[torch.Tensor] = None
        self.last_gate_activations: Optional[torch.Tensor] = None
        self.last_gated_activations: Optional[torch.Tensor] = None

    def _get_activation_functions(
        self, glu_variant: str
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        """Get activation and gate functions for the specified GLU variant.

        Args:
            glu_variant: GLU variant name.

        Returns:
            Tuple of (activation_function, gate_function).

        Raises:
            ValueError: If GLU variant is not supported.
        """
        variants = {
            "glu": (nn.Identity(), torch.sigmoid),
            "geglu": (F.gelu, nn.Identity()),
            "swiglu": (F.silu, nn.Identity()),  # SiLU is the same as Swish
        }

        if glu_variant not in variants:
            supported = ", ".join(variants.keys())
            raise ValueError(f"Unsupported GLU variant '{glu_variant}'. Supported: {supported}")

        return variants[glu_variant]

    def forward(self, x: torch.Tensor, store_activations: bool = False) -> torch.Tensor:
        """Apply gated feed-forward transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            store_activations: Whether to store intermediate activations for analysis.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(
                f"Input feature dimension {feature_dim} doesn't match d_model {self.d_model}"
            )

        # Compute main and gate activations
        main = self.linear_main(x)  # Shape: (batch_size, seq_len, d_ff)
        gate = self.linear_gate(x)  # Shape: (batch_size, seq_len, d_ff)

        # Apply activation functions
        main_activated = self.activation_fn(main)
        gate_activated = self.gate_fn(gate)

        # Apply gating mechanism
        gated = main_activated * gate_activated

        # Store activations for analysis if requested
        if store_activations:
            self.last_main_activations = main_activated.detach()
            self.last_gate_activations = gate_activated.detach()
            self.last_gated_activations = gated.detach()

        # Apply dropout
        gated = self.dropout(gated)

        # Final linear transformation
        output = self.linear_out(gated)  # Shape: (batch_size, seq_len, d_model)

        return output

    def get_activations(self) -> dict[str, Optional[torch.Tensor]]:
        """Get all stored activations for analysis.

        Returns:
            Dictionary with stored activation tensors.
        """
        return {
            "main_activations": self.last_main_activations,
            "gate_activations": self.last_gate_activations,
            "gated_activations": self.last_gated_activations,
        }

    def compute_activation_statistics(self) -> dict[str, float]:
        """Compute statistics about the stored activations.

        Returns:
            Dictionary with activation statistics for analysis.
        """
        stats = {}

        if self.last_main_activations is not None:
            main = self.last_main_activations
            stats.update({
                "main_activation_mean": main.mean().item(),
                "main_activation_std": main.std().item(),
                "main_activation_max": main.max().item(),
                "main_activation_min": main.min().item(),
            })

        if self.last_gate_activations is not None:
            gate = self.last_gate_activations
            stats.update({
                "gate_activation_mean": gate.mean().item(),
                "gate_activation_std": gate.std().item(),
                "gate_activation_max": gate.max().item(),
                "gate_activation_min": gate.min().item(),
            })

            # Gate-specific statistics
            if self.glu_variant == "glu":
                # For sigmoid gates, analyze how much gating is happening
                stats["gate_near_zero"] = (gate < 0.1).float().mean().item()
                stats["gate_near_one"] = (gate > 0.9).float().mean().item()

        if self.last_gated_activations is not None:
            gated = self.last_gated_activations
            stats.update({
                "gated_activation_mean": gated.mean().item(),
                "gated_activation_std": gated.std().item(),
                "gated_activation_max": gated.max().item(),
                "gated_activation_min": gated.min().item(),
                "gated_near_zero": (gated.abs() < 1e-6).float().mean().item(),
            })

        return stats

    def compute_gate_effectiveness(self) -> dict[str, float]:
        """Compute metrics about gate effectiveness.

        Returns:
            Dictionary with gate effectiveness metrics.
        """
        if self.last_main_activations is None or self.last_gate_activations is None:
            return {}

        main = self.last_main_activations
        gate = self.last_gate_activations
        gated = main * gate

        # Compute how much the gate changes the activations
        gate_effect = torch.abs(gated - main).mean()
        main_magnitude = torch.abs(main).mean()

        stats = {
            "gate_effect_ratio": (gate_effect / (main_magnitude + 1e-8)).item(),
            "gate_variance": gate.var().item(),
        }

        # Information flow analysis
        # How much information is passed vs. blocked
        if self.glu_variant == "glu":
            # For sigmoid gates
            info_passed = (gate > 0.5).float().mean().item()
            info_blocked = (gate < 0.1).float().mean().item()
            stats.update({
                "information_passed_ratio": info_passed,
                "information_blocked_ratio": info_blocked,
            })

        return stats

    def reset_parameters(self) -> None:
        """Reset all parameters to their initial values using Xavier initialization."""
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(self.linear_main.weight)
        nn.init.xavier_uniform_(self.linear_gate.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

        # Zero initialization for biases
        if self.linear_main.bias is not None:
            nn.init.zeros_(self.linear_main.bias)
        if self.linear_gate.bias is not None:
            nn.init.zeros_(self.linear_gate.bias)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"glu_variant={self.glu_variant}, dropout={self.dropout.p}"
        )
