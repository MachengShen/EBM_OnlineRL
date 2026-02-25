from __future__ import annotations

import torch
import torch.nn as nn

from .conditioning import apply_inpainting


class EquilibriumMatching1D(nn.Module):
    """EqM-style trajectory EBM over joint [s,a] packed trajectories [B, H+1, D]."""

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        transition_dim: int,
        action_dim: int,
        n_eqm_steps: int = 50,
        step_size: float = 0.05,
        c_scale: float = 1.0,
        clip_sample: bool = True,
        action_weight: float = 10.0,
        loss_discount: float = 1.0,
        obs_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.horizon = int(horizon)
        self.sequence_length = self.horizon + 1
        self.transition_dim = int(transition_dim)
        self.action_dim = int(action_dim)
        if self.action_dim <= 0 or self.action_dim >= self.transition_dim:
            raise ValueError(
                f"action_dim must be in [1, transition_dim-1], got action_dim={self.action_dim}, "
                f"transition_dim={self.transition_dim}"
            )

        self.n_eqm_steps = int(n_eqm_steps)
        self.step_size = float(step_size)
        self.c_scale = float(c_scale)
        self.clip_sample = bool(clip_sample)

        self.action_weight = float(action_weight)
        self.loss_discount = float(loss_discount)
        self.obs_loss_weight = float(obs_loss_weight)
        self.register_buffer("loss_weights", self._build_loss_weights())

    def _build_loss_weights(self) -> torch.Tensor:
        # Match GaussianDiffusion1D weighting for fair comparisons.
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        dim_weights[self.action_dim :] *= self.obs_loss_weight
        discounts = self.loss_discount ** torch.arange(self.sequence_length, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights[0, : self.action_dim] = self.action_weight
        return loss_weights

    def _sample_gamma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Keep shape [B, 1, 1] so broadcasting over [B, T, D] is unambiguous.
        return torch.rand((batch_size, 1, 1), device=device, dtype=dtype)

    def _c_of_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        # Safe default schedule: c(1)=0, c(0)=c_scale.
        return self.c_scale * (1.0 - gamma)

    def loss(self, x_start: torch.Tensor) -> torch.Tensor:
        if x_start.ndim != 3:
            raise ValueError(f"x_start must be [B,T,D], got {x_start.shape}")

        bsz = x_start.shape[0]
        eps = torch.randn_like(x_start)
        gamma = self._sample_gamma(bsz, device=x_start.device, dtype=x_start.dtype)
        c_gamma = self._c_of_gamma(gamma)

        x_gamma = gamma * x_start + (1.0 - gamma) * eps
        target = (eps - x_start) * c_gamma

        # EqM is time-invariant; feed constant timestep.
        t0 = torch.zeros((bsz,), device=x_start.device, dtype=torch.long)
        pred = self.model(x_gamma, t0)

        per_elem = (pred - target) ** 2
        return torch.mean(per_elem * self.loss_weights)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        obs0: torch.Tensor,
        goal: torch.Tensor,
        obs_dim: int,
        act_dim: int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        shape = (int(batch_size), self.sequence_length, self.transition_dim)
        x = torch.randn(shape, device=device, dtype=dtype)
        x = apply_inpainting(x, obs0=obs0, goal=goal, obs_dim=obs_dim, act_dim=act_dim)

        t0 = torch.zeros((int(batch_size),), device=device, dtype=torch.long)
        for _ in range(self.n_eqm_steps):
            grad = self.model(x, t0)
            x = x - self.step_size * grad
            if self.clip_sample:
                x = x.clamp(-1.0, 1.0)
            x = apply_inpainting(x, obs0=obs0, goal=goal, obs_dim=obs_dim, act_dim=act_dim)

        return x
