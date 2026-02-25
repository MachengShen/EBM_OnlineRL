#!/usr/bin/env python3
"""Shared utilities for Maze2D EqM research analysis scripts.

Usage: all analysis scripts must be run with the diffuser venv:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/<script_name>.py [args]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import gym
import numpy as np
import torch

import d4rl  # noqa: F401
from diffuser.datasets.sequence import GoalDataset
from diffuser.models.helpers import apply_conditioning
from diffuser.models.temporal import TemporalUnet

# Ensure probe script is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "scripts"))
from synthetic_maze2d_diffuser_probe import (
    EquilibriumMatchingDiffusion,
    SyntheticDatasetEnv,
    count_episodes_from_timeouts,
    normalize_condition,
    resolve_waypoint_t,
    sample_eval_waypoint,
)

# Maze2D constants
OBS_DIM = 4  # x, y, vx, vy
ACT_DIM = 2  # force_x, force_y
TRANSITION_DIM = ACT_DIM + OBS_DIM  # 6; packing: [act | obs]


def load_eqm_model_and_dataset(
    checkpoint_path: str,
    device: torch.device,
    weights: str = "ema",
) -> Tuple[EquilibriumMatchingDiffusion, GoalDataset, Dict]:
    """Load trained EqM model and matching GoalDataset from a checkpoint.

    Returns (model, dataset, config_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Parse dim_mults
    dim_mults = tuple(int(x) for x in str(cfg.get("model_dim_mults", "1,2,4")).split(","))

    # Build TemporalUnet (raw denoiser)
    denoiser = TemporalUnet(
        horizon=int(cfg["horizon"]),
        transition_dim=TRANSITION_DIM,
        cond_dim=OBS_DIM,
        dim=int(cfg.get("model_dim", 64)),
        dim_mults=dim_mults,
    )

    # Wrap in EqM
    model = EquilibriumMatchingDiffusion(
        model=denoiser,
        horizon=int(cfg["horizon"]),
        observation_dim=OBS_DIM,
        action_dim=ACT_DIM,
        n_eqm_steps=int(cfg.get("eqm_steps", 25)),
        step_size=float(cfg.get("eqm_step_size", 0.1)),
        c_scale=float(cfg.get("eqm_c_scale", 1.0)),
        clip_denoised=bool(cfg.get("eqm_clip_sample", True)),
        action_weight=1.0,
        loss_discount=1.0,
        obs_loss_weight=float(cfg.get("eqm_loss_obs_weight", 1.0)),
        loss_type="l2",
    ).to(device)

    # Load weights
    key = weights if weights in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    # Build matching GoalDataset
    env_name = str(cfg.get("env", "maze2d-umaze-v1"))
    horizon = int(cfg["horizon"])
    episode_len = int(cfg.get("episode_len", 256))
    max_path_length = int(cfg.get("max_path_length", 256))

    env = gym.make(env_name)
    raw_dataset = env.get_dataset()
    n_episodes = max(1, count_episodes_from_timeouts(raw_dataset["timeouts"]))

    synthetic_env = SyntheticDatasetEnv(
        name=f"{env_name}-synthetic-replay",
        dataset=raw_dataset,
        max_episode_steps=max(episode_len, 256),
    )
    dataset = GoalDataset(
        env=synthetic_env,
        horizon=horizon,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=max(max_path_length, 256),
        max_n_episodes=n_episodes + 16,
        use_padding=False,
    )

    return model, dataset, cfg


def get_replay_observations(dataset: GoalDataset) -> np.ndarray:
    """Extract un-normalized replay observations for waypoint sampling.

    Filters out zero-padded entries from episodes shorter than the buffer's
    max_path_length to avoid out-of-range observations (e.g. [0,0,0,0]).
    """
    obs = np.asarray(dataset.fields.observations[:, :, :], dtype=np.float32).reshape(-1, OBS_DIM)
    # Filter out zero-padded observations: valid maze2d positions are always > 0
    valid_mask = np.any(obs[:, :2] > 0, axis=1)
    return obs[valid_mask]


def eqm_sample_with_pos_only_waypoint(
    model: EquilibriumMatchingDiffusion,
    cond: Dict[int, torch.Tensor],
    waypoint_t: int,
    waypoint_xy_norm: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Run EqM sampling with position-only waypoint constraint.

    Standard start/goal conditioning via cond dict (writes all obs dims).
    Waypoint constrains ONLY the xy position dims (action_dim:action_dim+2),
    leaving velocity dims free.

    Args:
        model: EqM model
        cond: conditioning dict {0: start_obs_norm, H-1: goal_obs_norm}
        waypoint_t: timestep index for waypoint
        waypoint_xy_norm: normalized [x, y] tensor, shape [B, 2] or [2]
        horizon: trajectory length H

    Returns:
        Sampled trajectory [B, H, transition_dim]
    """
    batch_size = len(cond[0])
    device = next(model.parameters()).device

    x = torch.randn((batch_size, horizon, model.transition_dim), device=device)
    x = apply_conditioning(x, cond, model.action_dim)

    # Apply position-only waypoint
    wp = waypoint_xy_norm.to(device)
    if wp.ndim == 1:
        wp = wp.unsqueeze(0).expand(batch_size, -1)
    x[:, waypoint_t, ACT_DIM: ACT_DIM + 2] = wp

    t0 = torch.zeros((batch_size,), device=device, dtype=torch.long)
    for _ in range(max(1, model.n_eqm_steps)):
        grad = model.model(x, cond, t0)
        x = x - float(model.step_size) * grad
        if model.clip_denoised:
            x = torch.clamp(x, -1.0, 1.0)
        # Re-apply start/goal conditioning
        x = apply_conditioning(x, cond, model.action_dim)
        # Re-apply position-only waypoint (only xy, not velocity)
        x[:, waypoint_t, ACT_DIM: ACT_DIM + 2] = wp

    return x
