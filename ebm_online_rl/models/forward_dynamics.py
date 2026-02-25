"""Learned 1-step forward dynamics model for surrogate MPC alignment (H1).

Predicts s_{t+1} given (s_t, a_t) via residual connection:
    s_next_pred = s_t + net(concat[s_t, a_t])

This model operates in NORMALIZED trajectory space (same coordinate system
as the diffusion/EqM model) to ensure gradient alignment measurements
are meaningful.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ForwardDynamics(nn.Module):
    """Simple MLP forward dynamics model with residual prediction."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, obs_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next state from current state and action.

        Args:
            s: [B, obs_dim] or [B, T, obs_dim]
            a: [B, act_dim] or [B, T, act_dim]

        Returns:
            s_next: same shape as s
        """
        delta = self.net(torch.cat([s, a], dim=-1))
        return s + delta
