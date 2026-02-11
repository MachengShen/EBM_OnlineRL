from __future__ import annotations

import torch


def _expand_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.repeat(batch_size, 1)
    return x


def apply_inpainting(
    traj: torch.Tensor,
    obs0: torch.Tensor,
    goal: torch.Tensor,
    obs_dim: int,
    act_dim: int,
    clamp_last_action_zero: bool = True,
) -> torch.Tensor:
    """
    In-place conditioning:
    - traj[:, 0, :obs_dim] = obs0
    - traj[:, -1, :obs_dim] = goal
    - optionally traj[:, -1, obs_dim:obs_dim+act_dim] = 0
    """
    if traj.ndim != 3:
        raise ValueError(f"traj must be [B, H+1, D], got {traj.shape}")

    bsz = traj.shape[0]
    obs0 = _expand_batch(obs0, bsz).to(device=traj.device, dtype=traj.dtype)
    goal = _expand_batch(goal, bsz).to(device=traj.device, dtype=traj.dtype)

    traj[:, 0, :obs_dim] = obs0
    traj[:, -1, :obs_dim] = goal
    if clamp_last_action_zero:
        traj[:, -1, obs_dim : obs_dim + act_dim] = 0.0
    return traj

