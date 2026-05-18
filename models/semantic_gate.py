"""
Per-layer learnable semantic gates

This module implements learnable gates that control the influence of semantic
vectors at each transformer layer. The gates use sigmoid activation to stay
in the (0,1) range.
"""

import math

import torch
import torch.nn as nn


class PerLayerSemanticGate(nn.Module):
    """
    Learnable gate per transformer layer for semantic influence.

    Each layer gets its own gate value that determines how much semantic
    information is mixed with the token embeddings. The gates use sigmoid
    activation to naturally stay in (0,1) range.

    Attributes:
        num_layers (int): Number of transformer layers.
        raw_gates (nn.Parameter): Raw parameters before sigmoid activation.
    """

    def __init__(self, num_layers: int, init_gate: float = 0.3):
        """
        Initialize per-layer semantic gates.

        Args:
            num_layers: Number of transformer layers.
            init_gate: Initial gate value (0-1). Defaults to 0.3.
        """
        super().__init__()
        self.num_layers = num_layers

        # raw value where sigmoid(raw) = init_gate
        # sigmoid(0) = 0.5, sigmoid(-0.847) ≈ 0.3
        raw_init = math.log(init_gate / (1 - init_gate))
        self.raw_gates = nn.Parameter(torch.full((num_layers,), raw_init))

    def forward(self) -> torch.Tensor:
        """
        Return gates as values in (0,1) range.

        Returns:
            Tensor of gate values, one per layer.
        """
        return torch.sigmoid(self.raw_gates)

    def get_gate(self, layer_idx: int) -> float:
        """
        Get gate value for a specific layer.

        Args:
            layer_idx: Index of the layer.

        Returns:
            Gate value as a float.
        """
        return torch.sigmoid(self.raw_gates[layer_idx]).item()

    def set_gate(self, layer_idx: int, value: float) -> None:
        """
        Set gate for a specific layer to target value.

        Args:
            layer_idx: Index of the layer.
            value: Target gate value (clamped to 0.01-0.99).
        """
        value = max(0.01, min(0.99, value))
        raw_target = math.log(value / (1 - value))
        with torch.no_grad():
            self.raw_gates[layer_idx].fill_(raw_target)

    def get_all_gates(self) -> list:
        """
        Get all gate values as a list.

        Returns:
            List of gate values for all layers.
        """
        return torch.sigmoid(self.raw_gates).detach().cpu().tolist()

    def set_all_gates(self, values: list) -> None:
        """
        Set all gates from a list of values.

        Args:
            values: List of gate values for each layer.
        """
        for i, v in enumerate(values[: self.num_layers]):
            self.set_gate(i, v)


class SemanticGate(nn.Module):
    """
    Single learnable gate for semantic influence (legacy).

    This is a simpler single-gate version maintained for backward compatibility.

    Attributes:
        raw (nn.Parameter): Raw parameter before sigmoid activation.
    """

    def __init__(self, init_gate: float = 0.3):
        """
        Initialize a single semantic gate.

        Args:
            init_gate: Initial gate value (0-1). Defaults to 0.3.
        """
        super().__init__()
        raw_init = math.log(init_gate / (1 - init_gate))
        self.raw = nn.Parameter(torch.tensor(raw_init))

    def forward(self) -> torch.Tensor:
        """Return the gate value."""
        return torch.sigmoid(self.raw)

    def value(self) -> float:
        """Get the current gate value as a float."""
        return torch.sigmoid(self.raw).item()

    def set_value(self, target: float) -> None:
        """
        Set the gate to a target value.

        Args:
            target: Target gate value (clamped to 0.01-0.99).
        """
        target = max(0.01, min(0.99, target))
        raw_target = math.log(target / (1 - target))
        with torch.no_grad():
            self.raw.fill_(raw_target)
