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
from diffuser.models.diffusion import GaussianDiffusion
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


def load_diffuser_model_and_dataset(
    checkpoint_path: str,
    device: torch.device,
    weights: str = "ema",
) -> Tuple[GaussianDiffusion, GoalDataset, Dict]:
    """Load trained DDPM diffuser model and matching GoalDataset from a checkpoint.

    Returns (model, dataset, config_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    dim_mults = tuple(int(x) for x in str(cfg.get("model_dim_mults", "1,2,4")).split(","))

    denoiser = TemporalUnet(
        horizon=int(cfg["horizon"]),
        transition_dim=TRANSITION_DIM,
        cond_dim=OBS_DIM,
        dim=int(cfg.get("model_dim", 64)),
        dim_mults=dim_mults,
    )

    model = GaussianDiffusion(
        model=denoiser,
        horizon=int(cfg["horizon"]),
        observation_dim=OBS_DIM,
        action_dim=ACT_DIM,
        n_timesteps=int(cfg.get("n_diffusion_steps", 64)),
        clip_denoised=bool(cfg.get("clip_denoised", True)),
        predict_epsilon=bool(cfg.get("predict_epsilon", True)),
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
    ).to(device)

    key = weights if weights in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

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


def get_particle_offset(env) -> np.ndarray:
    """Return the (x, y) world-frame offset of the particle body.

    d4rl maze2d observations are slide-joint displacements.  The world
    position of the ball is:  world_xy = obs_xy + particle_body_pos[:2]
    """
    try:
        bid = env.sim.model.body_name2id('particle')
        return env.sim.model.body_pos[bid, :2].copy()
    except Exception:
        return np.array([1.2, 1.2])


def draw_maze_walls(ax, env, obs_offset=None):
    """Draw maze walls in observation (joint-displacement) coordinate space."""
    import matplotlib.patches as mpatches
    try:
        env.sim.forward()
        mj_model = env.sim.model
        mj_data  = env.sim.data
        if obs_offset is None:
            obs_offset = get_particle_offset(env)
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] != 6:
                continue
            size = mj_model.geom_size[i]
            if size[0] > 2.0 or size[1] > 2.0:
                continue
            if size[0] < 0.05 or size[1] < 0.05:
                continue
            wpos = mj_data.geom_xpos[i]
            ox = wpos[0] - obs_offset[0]
            oy = wpos[1] - obs_offset[1]
            rect = mpatches.Rectangle(
                (ox - size[0], oy - size[1]),
                2.0 * size[0], 2.0 * size[1],
                linewidth=0.5, edgecolor="#333333",
                facecolor="#666666", alpha=0.45, zorder=0,
            )
            ax.add_patch(rect)
    except Exception as e:
        print(f"  [warn] draw_maze_walls: {e}")


def build_wall_rects(env, obs_offset):
    """Return list of (cx, cy, half_w, half_h) in obs-frame coordinates."""
    env.sim.forward()
    mj_model = env.sim.model
    mj_data  = env.sim.data
    rects = []
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] != 6:
            continue
        size = mj_model.geom_size[i]
        if size[0] > 2.0 or size[1] > 2.0:
            continue
        if size[0] < 0.05 or size[1] < 0.05:
            continue
        wpos = mj_data.geom_xpos[i]
        cx = wpos[0] - obs_offset[0]
        cy = wpos[1] - obs_offset[1]
        rects.append((cx, cy, size[0], size[1]))
    return rects


def min_wall_surface_dist(xy, wall_rects) -> float:
    """Minimum distance from point xy to the nearest wall surface."""
    min_d = float("inf")
    for cx, cy, hw, hh in wall_rects:
        dx = max(abs(xy[0] - cx) - hw, 0.0)
        dy = max(abs(xy[1] - cy) - hh, 0.0)
        d = np.sqrt(dx * dx + dy * dy)
        min_d = min(min_d, d)
    return min_d


def filter_replay_by_wall_dist(replay_obs, wall_rects, min_dist):
    """Return indices of replay observations that are >= min_dist from walls."""
    xy = replay_obs[:, :2]
    valid = [i for i in range(len(xy))
             if min_wall_surface_dist(xy[i], wall_rects) >= min_dist]
    return np.array(valid, dtype=int)


def sample_start_goal_from_replay(replay_obs, rng, min_dist=1.0,
                                   wall_rects=None, min_wall_dist=0.0,
                                   max_attempts=500):
    """Sample start/goal from replay with optional wall-distance rejection."""
    xy = replay_obs[:, :2]
    for _ in range(max_attempts):
        idx_s = rng.integers(len(xy))
        idx_g = rng.integers(len(xy))
        if np.linalg.norm(xy[idx_s] - xy[idx_g]) < min_dist:
            continue
        if wall_rects and min_wall_dist > 0:
            if (min_wall_surface_dist(xy[idx_s], wall_rects) < min_wall_dist or
                    min_wall_surface_dist(xy[idx_g], wall_rects) < min_wall_dist):
                continue
        return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()
    # Fallback: farthest valid pair
    if wall_rects and min_wall_dist > 0:
        valid_idx = filter_replay_by_wall_dist(replay_obs, wall_rects, min_wall_dist)
        if len(valid_idx) > 1:
            xy_v = xy[valid_idx]
            idx_s = rng.integers(len(xy_v))
            dists = np.linalg.norm(xy_v - xy_v[idx_s][None, :], axis=1)
            idx_g = int(np.argmax(dists))
            return replay_obs[valid_idx[idx_s]].copy(), replay_obs[valid_idx[idx_g]].copy()
    idx_s = rng.integers(len(xy))
    dists = np.linalg.norm(xy - xy[idx_s][None, :], axis=1)
    idx_g = int(np.argmax(dists))
    return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()


def sample_on_path_waypoint(replay_obs, start_xy, goal_xy, rng,
                             wall_rects=None, min_wall_dist=0.0):
    """Sample a waypoint near the geometric midpoint of start→goal."""
    midpoint = (start_xy + goal_xy) / 2.0
    pos = replay_obs[:, :2]
    dists = np.linalg.norm(pos - midpoint[None, :], axis=1)
    k = min(50, len(pos))
    nearest_k = np.argsort(dists)[:k]
    if wall_rects and min_wall_dist > 0:
        valid = [j for j in nearest_k
                 if min_wall_surface_dist(pos[j], wall_rects) >= min_wall_dist]
        if valid:
            nearest_k = np.array(valid)
    chosen = rng.choice(nearest_k)
    return pos[chosen].copy()


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
