from __future__ import annotations

import numpy as np
import torch

from .diffusion import GaussianDiffusion1D


@torch.no_grad()
def plan_action(
    model: GaussianDiffusion1D,
    obs: np.ndarray,
    goal: np.ndarray,
    obs_dim: int,
    act_dim: int,
    action_scale: float,
    device: torch.device,
    n_samples: int = 1,
    check_conditioning: bool = True,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)

    traj = model.sample(
        batch_size=n_samples,
        obs0=obs_t,
        goal=goal_t,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    if check_conditioning:
        obs_batch = obs_t.unsqueeze(0).repeat(n_samples, 1)
        goal_batch = goal_t.unsqueeze(0).repeat(n_samples, 1)
        ok_start = torch.allclose(traj[:, 0, :obs_dim], obs_batch, atol=1e-5)
        ok_goal = torch.allclose(traj[:, -1, :obs_dim], goal_batch, atol=1e-5)
        if not (ok_start and ok_goal):
            raise RuntimeError("Inpainting check failed: sampled trajectory does not satisfy clamped start/goal.")

    action_norm = traj[0, 0, obs_dim : obs_dim + act_dim]
    action = action_norm * action_scale
    action = torch.clamp(action, min=-action_scale, max=action_scale)
    return action.detach().cpu().numpy().astype(np.float32)

