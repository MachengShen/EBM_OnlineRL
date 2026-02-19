from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .conditioning import apply_inpainting


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    steps = n_timesteps + 1
    x = torch.linspace(0, steps, steps, dtype=dtype)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion1D(nn.Module):
    """
    Diffusion over [B, H+1, transition_dim] trajectories.

    This model is trained unconditionally and planned with inpainting
    (clamp start and goal states each denoising step).
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        transition_dim: int,
        action_dim: int,
        n_diffusion_steps: int = 32,
        predict_epsilon: bool = True,
        clip_denoised: bool = True,
        action_weight: float = 1.0,
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
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.predict_epsilon = bool(predict_epsilon)
        self.clip_denoised = bool(clip_denoised)
        self.action_weight = float(action_weight)
        self.loss_discount = float(loss_discount)
        self.obs_loss_weight = float(obs_loss_weight)

        betas = cosine_beta_schedule(self.n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer("loss_weights", self._build_loss_weights())

    def _build_loss_weights(self) -> torch.Tensor:
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        dim_weights[self.action_dim :] *= self.obs_loss_weight

        discounts = self.loss_discount ** torch.arange(self.sequence_length, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights[0, : self.action_dim] = self.action_weight
        return loss_weights

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.predict_epsilon:
            return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
            ) * noise
        return noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_out = self.model(x_t, t)
        x_recon = self.predict_start_from_noise(x_t, t, model_out)
        if self.clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x_t, t=t)

    def loss(self, x_start: torch.Tensor) -> torch.Tensor:
        bsz = x_start.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (bsz,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)
        if self.predict_epsilon:
            per_elem_loss = (model_out - noise) ** 2
        else:
            per_elem_loss = (model_out - x_start) ** 2
        return torch.mean(per_elem_loss * self.loss_weights)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        obs0: torch.Tensor,
        goal: torch.Tensor,
        obs_dim: int,
        act_dim: int,
    ) -> torch.Tensor:
        device = self.betas.device
        shape = (batch_size, self.sequence_length, self.transition_dim)
        x = torch.randn(shape, device=device)
        x = apply_inpainting(x, obs0=obs0, goal=goal, obs_dim=obs_dim, act_dim=act_dim)

        for i in reversed(range(self.n_diffusion_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            model_mean, _, model_log_variance = self.p_mean_variance(x, t)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            x = apply_inpainting(x, obs0=obs0, goal=goal, obs_dim=obs_dim, act_dim=act_dim)

        return x
