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
    check_conditioning: bool = False,
    control_mode: str = "action",
    double_integrator_dt: float = 1.0,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)

    traj = model.sample(
        batch_size=1,
        obs0=obs_t,
        goal=goal_t,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    if check_conditioning:
        ok_start = torch.allclose(traj[0, 0, :obs_dim], obs_t, atol=1e-5)
        ok_goal = torch.allclose(traj[0, -1, :obs_dim], goal_t, atol=1e-5)
        if not (ok_start and ok_goal):
            raise RuntimeError("Inpainting check failed: sampled trajectory does not satisfy clamped start/goal.")

    if control_mode == "action":
        action_norm = traj[0, 0, obs_dim : obs_dim + act_dim]
        action = action_norm * action_scale
    elif control_mode == "waypoint":
        next_state = traj[0, 1, :obs_dim]
        if obs_dim == act_dim:
            action = next_state - obs_t
        elif obs_dim == 2 * act_dim:
            dt = max(float(double_integrator_dt), 1e-6)
            vel = obs_t[act_dim : 2 * act_dim]
            vel_next = next_state[act_dim : 2 * act_dim]
            action = (vel_next - vel) / dt
        else:
            raise ValueError(
                "Waypoint control expects either first-order (obs_dim==act_dim) or "
                "double-integrator (obs_dim==2*act_dim) dynamics."
            )
        # Keep planner actions within the same physical action limit used by random exploration.
        action = torch.clamp(action, -action_scale, action_scale)
    else:
        raise ValueError(f"Unknown control_mode={control_mode}. Expected one of: action, waypoint.")
    return action.detach().cpu().numpy().astype(np.float32)
