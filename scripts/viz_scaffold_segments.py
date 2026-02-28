#!/usr/bin/env python3
"""Visualize three aspects of the scaffold EqM training:
  1. Planned h=64 trajectory segments (model output)
  2. Replay buffer (d4rl offline) episode trajectories
  3. Medium and large maze layouts (walls only)

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype:/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/viz_scaffold_segments.py \
    --checkpoint <path/to/checkpoint.pt> \
    --out_dir /tmp/viz_scaffold_segments
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gym
import numpy as np
import torch

import d4rl  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    OBS_DIM, ACT_DIM,
    get_replay_observations,
    load_eqm_model_and_dataset,
)
from synthetic_maze2d_diffuser_probe import (
    count_episodes_from_timeouts,
    SyntheticDatasetEnv,
)
from diffuser.datasets.sequence import GoalDataset
from diffuser.models.helpers import apply_conditioning


# ─────────────────────────────────────────────────────────────────────────────
# Wall rendering utilities (copied from viz_maze2d_waypoint_exec_trajectories)
# ─────────────────────────────────────────────────────────────────────────────

def get_particle_offset(env):
    try:
        bid = env.sim.model.body_name2id('particle')
        return env.sim.model.body_pos[bid, :2].copy()
    except Exception:
        return np.array([1.2, 1.2])


def draw_maze_walls(ax, env, obs_offset=None, alpha=0.45):
    try:
        env.sim.forward()
        mj_model = env.sim.model
        mj_data = env.sim.data
        if obs_offset is None:
            obs_offset = get_particle_offset(env)
        drawn = 0
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
                facecolor="#666666", alpha=alpha, zorder=0,
            )
            ax.add_patch(rect)
            drawn += 1
        return obs_offset
    except Exception as e:
        print(f"  [warn] draw_maze_walls: {e}")
        return np.array([1.2, 1.2])


def setup_ax(ax, title, obs_range):
    lo, hi = obs_range
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Planned trajectory segments
# ─────────────────────────────────────────────────────────────────────────────

def fig_planned_segments(model, dataset, env, cfg, rng, n_panels=8, out_path=None):
    horizon = int(cfg["horizon"])
    action_dim = ACT_DIM

    replay_obs = get_replay_observations(dataset)
    obs_xy = replay_obs[:, :2]
    obs_offset = get_particle_offset(env)

    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f"EqM planned trajectory segments  (h={horizon}, step=4k offline init)", fontsize=11)

    # Estimate obs coordinate range from replay
    xy_min = obs_xy.min(0) - 0.3
    xy_max = obs_xy.max(0) + 0.3
    obs_range = (min(xy_min[0], xy_min[1]) - 0.1, max(xy_max[0], xy_max[1]) + 0.1)

    device = next(model.parameters()).device

    for panel_i in range(n_panels):
        ax = axes[panel_i]
        draw_maze_walls(ax, env, obs_offset)
        setup_ax(ax, f"Plan {panel_i+1}", obs_range)

        # Sample a start/goal pair well separated
        for _ in range(200):
            idx_s = rng.integers(len(replay_obs))
            idx_g = rng.integers(len(replay_obs))
            d = np.linalg.norm(obs_xy[idx_s] - obs_xy[idx_g])
            if d > 1.0:
                break
        start_obs = replay_obs[idx_s].copy()
        goal_obs = replay_obs[idx_g].copy()

        # Normalize and build condition: {0: start_norm [1,4], H-1: goal_norm [1,4]}
        from synthetic_maze2d_diffuser_probe import normalize_condition as _nc
        start_t = _nc(dataset, start_obs, device)  # [1, OBS_DIM]
        goal_t  = _nc(dataset, goal_obs,  device)  # [1, OBS_DIM]
        cond = {0: start_t, horizon - 1: goal_t}

        # Run EqM inference
        with torch.no_grad():
            traj = model.conditional_sample(cond, horizon=horizon)  # [1, H, T_dim]

        traj_np = traj[0].cpu().numpy()  # [H, T_dim]: [act(2) | obs(4)]

        # Unnormalize: obs lives at dims [action_dim : action_dim+OBS_DIM]
        obs_norm = traj_np[:, action_dim: action_dim + OBS_DIM]  # [H, 4]
        try:
            obs_real = dataset.normalizer.unnormalize(obs_norm, "observations")  # [H, 4]
        except Exception:
            obs_real = obs_norm  # fallback: use normalized values directly
        plan_xy_real = obs_real[:, :2]  # [H, 2]: x, y

        # Mark scaffold insertion points (every scaffold_stride within middle frac)
        stride = 8
        insert_frac = 0.3
        n_mid = max(1, int(horizon * insert_frac))
        mid_start = (horizon - n_mid) // 2
        scaffold_steps = list(range(mid_start, mid_start + n_mid, stride))

        ax.plot(plan_xy_real[:, 0], plan_xy_real[:, 1],
                "b-", linewidth=1.2, alpha=0.8, zorder=2)
        ax.scatter(*start_obs[:2], c="green", s=60, zorder=5, marker="o", label="start")
        ax.scatter(*goal_obs[:2], c="red", s=60, zorder=5, marker="*", label="goal")

        # Scaffold insertion points
        sc_pts = plan_xy_real[scaffold_steps] if len(scaffold_steps) > 0 else np.zeros((0, 2))
        if len(sc_pts):
            ax.scatter(sc_pts[:, 0], sc_pts[:, 1],
                       c="orange", s=25, zorder=4, marker="^", alpha=0.8)

        if panel_i == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused panels
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Add legend for scaffold markers
    fig.text(0.5, 0.01,
             "Blue line = planned trajectory  |  ▲ orange = scaffold anchor insertion points",
             ha="center", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = out_path or "/tmp/viz_scaffold_segments/planned_segments.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Replay buffer (d4rl offline) episode trajectories
# ─────────────────────────────────────────────────────────────────────────────

def fig_replay_buffer(env, env_name, obs_offset, rng, n_episodes=40, out_path=None):
    """Show offline d4rl episode trajectories (the initial replay buffer data)."""
    raw = env.get_dataset()
    obs_all = raw["observations"]  # [N, 4]: x, y, vx, vy
    timeouts = raw["timeouts"].astype(bool)
    terminals = raw["terminals"].astype(bool)
    episode_ends = np.where(timeouts | terminals)[0]

    # Reconstruct episode slices
    starts = np.concatenate([[0], episode_ends[:-1] + 1])
    ends = episode_ends + 1
    n_ep_total = len(starts)
    print(f"  d4rl {env_name}: {n_ep_total} episodes, {len(obs_all)} transitions")

    # Sample up to n_episodes
    chosen = rng.choice(n_ep_total, size=min(n_episodes, n_ep_total), replace=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(f"d4rl offline replay trajectories ({env_name})\n"
                 f"(initial training buffer, {min(n_episodes, n_ep_total)} of {n_ep_total} episodes)",
                 fontsize=10)

    draw_maze_walls(ax, env, obs_offset)

    xy_all = obs_all[:, :2]
    obs_range = (xy_all.min() - 0.3, xy_all.max() + 0.3)
    setup_ax(ax, "", obs_range)

    cmap = cm.get_cmap("tab20", len(chosen))
    for color_i, ep_i in enumerate(chosen):
        s, e = starts[ep_i], ends[ep_i]
        ep_xy = obs_all[s:e, :2]
        color = cmap(color_i)
        ax.plot(ep_xy[:, 0], ep_xy[:, 1],
                "-", color=color, linewidth=0.6, alpha=0.5, zorder=2)
        # Mark start and end
        ax.scatter(*ep_xy[0], c=[color], s=20, marker="o", alpha=0.8, zorder=3)
        ax.scatter(*ep_xy[-1], c=[color], s=20, marker="x", alpha=0.8, zorder=3)

    ax.set_title(f"d4rl offline replay trajectories — {env_name}", fontsize=9)
    fig.text(0.5, 0.01, "Each color = one episode  |  ● start  × end", ha="center", fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = out_path or "/tmp/viz_scaffold_segments/replay_buffer.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Medium and large maze layouts
# ─────────────────────────────────────────────────────────────────────────────

def fig_maze_layouts(out_path=None):
    """Render the wall structure of umaze, medium, and large mazes side by side."""
    maze_names = [
        ("maze2d-umaze-v1",  "U-maze (umaze)"),
        ("maze2d-medium-v1", "Medium maze"),
        ("maze2d-large-v1",  "Large maze"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Maze2D environment layouts (wall geometry)", fontsize=12)

    for ax, (env_name, title) in zip(axes, maze_names):
        print(f"  Loading {env_name}...")
        env = gym.make(env_name)
        env.reset()

        obs_offset = get_particle_offset(env)
        draw_maze_walls(ax, env, obs_offset, alpha=0.65)

        # Get dataset to determine obs range
        raw = env.get_dataset()
        xy = raw["observations"][:, :2]
        pad = 0.4
        xlo, xhi = xy[:, 0].min() - pad, xy[:, 0].max() + pad
        ylo, yhi = xy[:, 1].min() - pad, xy[:, 1].max() + pad
        lo = min(xlo, ylo)
        hi = max(xhi, yhi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)

        # Scatter a thin cloud of the dataset trajectories for free-space context
        step = max(1, len(xy) // 2000)
        ax.scatter(xy[::step, 0], xy[::step, 1],
                   s=0.5, c="steelblue", alpha=0.15, zorder=1)

        env.close()

    plt.tight_layout()
    out = out_path or "/tmp/viz_scaffold_segments/maze_layouts.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to EqM checkpoint .pt")
    ap.add_argument("--out_dir", default="/tmp/viz_scaffold_segments")
    ap.add_argument("--n_plan_panels", type=int, default=8)
    ap.add_argument("--n_replay_episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")  # CPU is fine for inference-only viz

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Loading EqM model + dataset ===")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    env_name = str(cfg.get("env", "maze2d-umaze-v1"))
    env = gym.make(env_name)
    env.reset()
    obs_offset = get_particle_offset(env)

    print("=== Figure 1: Planned trajectory segments ===")
    fig_planned_segments(
        model, dataset, env, cfg, rng,
        n_panels=args.n_plan_panels,
        out_path=str(out / "planned_segments.png"),
    )

    print("=== Figure 2: Replay buffer episode trajectories ===")
    fig_replay_buffer(
        env, env_name, obs_offset, rng,
        n_episodes=args.n_replay_episodes,
        out_path=str(out / "replay_buffer.png"),
    )

    print("=== Figure 3: Maze layouts (umaze / medium / large) ===")
    fig_maze_layouts(out_path=str(out / "maze_layouts.png"))

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
