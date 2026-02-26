#!/usr/bin/env python3
"""Visualize execution-level waypoint trajectories on Maze2D (v3).

Changes from v2:
- Fixed coordinate system: uses data.geom_xpos (world frame) minus particle
  body offset to align walls with observation (joint-displacement) space.
- Wall-distance rejection sampling: start/goal/waypoint positions must be at
  least --min_wall_dist from any wall surface.
- Default --n_episodes=16 for richer visual inspection.
- Outputs per-episode stats table in stdout.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \\
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \\
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \\
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \\
    scripts/viz_maze2d_waypoint_exec_trajectories.py \\
    --checkpoint <path> --n_episodes 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    eqm_sample_with_pos_only_waypoint,
    get_replay_observations,
    load_eqm_model_and_dataset,
    normalize_condition,
    sample_eval_waypoint,
)
from synthetic_maze2d_diffuser_probe import safe_reset, safe_step

import d4rl  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# Maze wall rendering
# ─────────────────────────────────────────────────────────────────────────────

def get_particle_offset(env):
    """Return the (x, y) world-frame offset of the particle body.

    In d4rl maze2d, observations are slide-joint displacements, NOT world
    positions.  The world position of the ball is:
        world_xy = obs_xy + particle_body_pos[:2]
    so to draw walls in observation coordinate space we subtract this offset.
    """
    try:
        bid = env.sim.model.body_name2id('particle')
        return env.sim.model.body_pos[bid, :2].copy()
    except Exception:
        return np.array([1.2, 1.2])  # fallback: value from d4rl source


def draw_maze_walls(ax, env, obs_offset=None):
    """Draw maze walls in observation (joint-displacement) coordinate space.

    d4rl maze2d observations are slide-joint displacements from the particle
    body's initial position.  Wall geoms are defined in world frame.  This
    function converts wall positions to observation space by subtracting the
    particle body offset before drawing.

    Filters box-type geoms with footprint half-size in (0.05, 2.0) to skip
    the floor plane and large boundary geoms.
    """
    try:
        env.sim.forward()
        mj_model = env.sim.model
        mj_data  = env.sim.data
        if obs_offset is None:
            obs_offset = get_particle_offset(env)
        drawn = 0
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] != 6:          # only box geoms
                continue
            size = mj_model.geom_size[i]
            if size[0] > 2.0 or size[1] > 2.0:     # skip large floor tiles
                continue
            if size[0] < 0.05 or size[1] < 0.05:   # skip tiny geoms
                continue
            # geom_xpos is world-frame centroid; convert to obs frame
            wpos = mj_data.geom_xpos[i]             # shape (3,): world x,y,z
            ox = wpos[0] - obs_offset[0]
            oy = wpos[1] - obs_offset[1]
            rect = mpatches.Rectangle(
                (ox - size[0], oy - size[1]),
                2.0 * size[0], 2.0 * size[1],
                linewidth=0.5,
                edgecolor="#333333",
                facecolor="#666666",
                alpha=0.45,
                zorder=0,
            )
            ax.add_patch(rect)
            drawn += 1
        return drawn > 0
    except Exception as e:
        print(f"  [warn] Could not draw maze walls: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Wall-distance utilities
# ─────────────────────────────────────────────────────────────────────────────

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


def min_wall_surface_dist(xy, wall_rects):
    """Minimum distance from point xy to the nearest wall surface.

    Returns 0 if the point is inside any wall.
    """
    min_d = float("inf")
    for cx, cy, hw, hh in wall_rects:
        # Distance from point to axis-aligned rectangle surface
        dx = max(abs(xy[0] - cx) - hw, 0.0)
        dy = max(abs(xy[1] - cy) - hh, 0.0)
        d = np.sqrt(dx * dx + dy * dy)
        min_d = min(min_d, d)
    return min_d


def filter_replay_by_wall_dist(replay_obs, wall_rects, min_dist):
    """Return indices of replay observations that are >= min_dist from walls."""
    xy = replay_obs[:, :2]
    valid = []
    for i in range(len(xy)):
        if min_wall_surface_dist(xy[i], wall_rects) >= min_dist:
            valid.append(i)
    return np.array(valid, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Waypoint sampling strategies (with wall-distance rejection)
# ─────────────────────────────────────────────────────────────────────────────

def sample_start_goal_from_replay(replay_obs, rng, min_dist=1.0,
                                  wall_rects=None, min_wall_dist=0.0,
                                  max_attempts=500):
    """Sample start/goal from replay, with optional wall-distance rejection."""
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
    # Fallback: pick the farthest pair from valid positions
    if wall_rects and min_wall_dist > 0:
        valid_idx = filter_replay_by_wall_dist(replay_obs, wall_rects, min_wall_dist)
        if len(valid_idx) > 1:
            xy_v = xy[valid_idx]
            idx_s = rng.integers(len(xy_v))
            dists = np.linalg.norm(xy_v - xy_v[idx_s][None, :], axis=1)
            idx_g = int(np.argmax(dists))
            return replay_obs[valid_idx[idx_s]].copy(), replay_obs[valid_idx[idx_g]].copy()
    # Ultimate fallback
    idx_s = rng.integers(len(xy))
    dists = np.linalg.norm(xy - xy[idx_s][None, :], axis=1)
    idx_g = int(np.argmax(dists))
    return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()


def sample_on_path_waypoint(replay_obs, start_xy, goal_xy, rng,
                            wall_rects=None, min_wall_dist=0.0):
    """Sample a waypoint near the geometric midpoint of start→goal.

    By picking the midpoint as target, we bias toward positions the agent
    would naturally pass through, avoiding large detours.  Optionally
    rejects candidates too close to walls.
    """
    midpoint = (start_xy + goal_xy) / 2.0
    pos = replay_obs[:, :2]
    dists = np.linalg.norm(pos - midpoint[None, :], axis=1)
    k = min(50, len(pos))
    nearest_k = np.argsort(dists)[:k]
    # Filter by wall distance if requested
    if wall_rects and min_wall_dist > 0:
        valid = [j for j in nearest_k
                 if min_wall_surface_dist(pos[j], wall_rects) >= min_wall_dist]
        if valid:
            nearest_k = np.array(valid)
    chosen = rng.choice(nearest_k)
    return pos[chosen].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────

def plan_from_obs(model, dataset, obs, goal_obs, device, horizon,
                  waypoint_xy=None, waypoint_t=None,
                  waypoint_mode="pos_only"):
    """Plan a single trajectory from obs to goal, optionally with a waypoint.

    Returns (actions, observations, used_waypoint_flag).
    """
    start_norm = normalize_condition(dataset, obs, device)        # (1, obs_dim)
    goal_norm  = normalize_condition(dataset, goal_obs, device)   # (1, obs_dim)
    cond = {0: start_norm, horizon - 1: goal_norm}

    use_wp = (
        waypoint_xy is not None
        and waypoint_t is not None
        and 1 <= waypoint_t <= horizon - 2
    )

    if use_wp and waypoint_mode == "pos_only":
        wp_obs = np.zeros(OBS_DIM, dtype=np.float32)
        wp_obs[:2] = waypoint_xy
        wp_full_norm = normalize_condition(dataset, wp_obs, device)  # (1, obs_dim)
        wp_xy_norm = wp_full_norm[:, :2]                             # (1, 2)
        traj = eqm_sample_with_pos_only_waypoint(
            model, cond, waypoint_t, wp_xy_norm, horizon)
    else:
        traj = model.conditional_sample(cond, horizon=horizon, verbose=False)

    # traj shape: (1, horizon, transition_dim)
    act_norm = traj[:, :, :ACT_DIM].detach().cpu().numpy()
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    actions      = dataset.normalizer.unnormalize(act_norm, "actions")[0]
    observations = dataset.normalizer.unnormalize(obs_norm, "observations")[0]
    return actions, observations, use_wp


# ─────────────────────────────────────────────────────────────────────────────
# Episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_with_trajectory(model, dataset, env, device, horizon,
                                 goal_xy, goal_obs, waypoint_xy,
                                 waypoint_mode="pos_only",
                                 replan_every=8, max_steps=300,
                                 eps_wp=0.2, eps_goal=0.5):
    """Run one episode; return trajectory + two key plan snapshots."""
    obs = safe_reset(env)
    start_xy = obs[:2].copy()
    t_wp_global = (horizon - 1) // 2
    k_exec = 0
    planned_actions = None
    plan_offset = 0

    trajectory = [obs[:2].copy()]

    # We save at most 2 plan snapshots:
    #   plan_with_wp    – first plan taken while waypoint constraint is active
    #   plan_without_wp – first plan taken AFTER the constraint drops
    plan_with_wp    = None
    plan_without_wp = None

    act_low  = env.action_space.low
    act_high = env.action_space.high
    min_wp_dist   = float("inf")
    min_goal_dist = float("inf")
    hit_wp   = False
    hit_goal = False

    for step in range(max_steps):
        need_replan = (
            planned_actions is None
            or step % replan_every == 0
            or plan_offset >= len(planned_actions)
        )

        if need_replan:
            t_wp_local = t_wp_global - k_exec
            with torch.no_grad():
                pa, po, used_wp = plan_from_obs(
                    model, dataset, obs, goal_obs, device, horizon,
                    waypoint_xy=waypoint_xy,
                    waypoint_t=t_wp_local,
                    waypoint_mode=waypoint_mode,
                )
            planned_actions = pa
            plan_offset = 0

            # Capture the first plan of each type
            if used_wp and plan_with_wp is None:
                plan_with_wp = po[:, :2].copy()
            if not used_wp and plan_without_wp is None:
                plan_without_wp = po[:, :2].copy()

        if plan_offset < len(planned_actions):
            action = planned_actions[plan_offset]
            plan_offset += 1
        else:
            action = np.zeros(ACT_DIM, dtype=np.float32)

        action = np.clip(action, act_low, act_high).astype(np.float32)
        obs, reward, done, info = safe_step(env, action)
        k_exec += 1
        trajectory.append(obs[:2].copy())

        if waypoint_xy is not None:
            wp_dist = float(np.linalg.norm(obs[:2] - waypoint_xy))
            min_wp_dist = min(min_wp_dist, wp_dist)
            if wp_dist <= eps_wp and not hit_wp:
                hit_wp = True

        goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = min(min_goal_dist, goal_dist)
        if goal_dist <= eps_goal and not hit_goal:
            hit_goal = True

        if done:
            break

    return {
        "trajectory":        np.array(trajectory),
        "plan_with_wp":      plan_with_wp,
        "plan_without_wp":   plan_without_wp,
        "start_xy":          start_xy,
        "goal_xy":           goal_xy,
        "waypoint_xy":       waypoint_xy,
        "min_wp_dist":       min_wp_dist,
        "min_goal_dist":     min_goal_dist,
        "hit_wp":            hit_wp,
        "hit_goal":          hit_goal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_episode(ep_data, ax, env, title="", eps_wp=0.2, obs_offset=None):
    """Plot a single episode: walls → plans → executed trajectory → markers."""
    # 1. Maze walls (background) — drawn in obs coordinate frame
    draw_maze_walls(ax, env, obs_offset=obs_offset)

    wp   = ep_data["waypoint_xy"]
    traj = ep_data["trajectory"]

    # 2. Planned trajectories (two snapshots)
    if ep_data["plan_with_wp"] is not None:
        p = ep_data["plan_with_wp"]
        ax.plot(p[:, 0], p[:, 1], "--", color="#00bb55", alpha=0.6,
                linewidth=1.4, label="plan (wp active)", zorder=2)
    if ep_data["plan_without_wp"] is not None:
        p = ep_data["plan_without_wp"]
        ax.plot(p[:, 0], p[:, 1], "--", color="#aaaaaa", alpha=0.6,
                linewidth=1.4, label="plan (no wp)", zorder=2)

    # 3. Executed trajectory
    ax.plot(traj[:, 0], traj[:, 1], "-", color="#2255cc", alpha=0.85,
            linewidth=1.8, label="executed", zorder=3)

    # 4. Start / goal markers
    ax.plot(ep_data["start_xy"][0], ep_data["start_xy"][1],
            "o", color="black", markersize=9, zorder=5, label="start")
    ax.plot(ep_data["goal_xy"][0], ep_data["goal_xy"][1],
            "*", color="red", markersize=14, zorder=5, label="goal")

    # 5. Waypoint: diamond + dashed radius circle + closest-approach cross
    if wp is not None:
        circle = mpatches.Circle(wp, eps_wp, fill=False, color="orange",
                                  linewidth=2.0, linestyle="--", zorder=4)
        ax.add_patch(circle)
        ax.plot(wp[0], wp[1], "D", color="orange", markersize=11, zorder=5,
                label=f"waypoint (eps={eps_wp})")
        # Closest approach to waypoint
        dists = np.linalg.norm(traj - wp[None, :], axis=1)
        idx = int(np.argmin(dists))
        ax.plot(traj[idx, 0], traj[idx, 1], "x", color="darkorange",
                markersize=12, markeredgewidth=3, zorder=6)

    # 6. Title / styling
    hit_str  = "HIT" if ep_data["hit_wp"]   else f"MISS ({ep_data['min_wp_dist']:.2f})"
    goal_str = "HIT" if ep_data["hit_goal"] else f"MISS ({ep_data['min_goal_dist']:.2f})"
    ax.set_title(f"{title}\nWP: {hit_str} | Goal: {goal_str}", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes",       type=int,   default=16)
    parser.add_argument("--replan_every",     type=int,   default=8)
    parser.add_argument("--max_steps",        type=int,   default=300)
    parser.add_argument("--eps_wp",           type=float, default=0.2,
                        help="Waypoint success radius (default: 0.2)")
    parser.add_argument("--min_wall_dist",    type=float, default=0.3,
                        help="Min distance from start/goal/wp to wall surface")
    parser.add_argument("--min_start_goal_dist", type=float, default=1.0,
                        help="Min Euclidean distance between start and goal")
    parser.add_argument("--waypoint_strategy",
                        choices=["feasible", "on_path"], default="on_path",
                        help=(
                            "'on_path': waypoint near start→goal midpoint (easier); "
                            "'feasible': random from replay (original)"
                        ))
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or f"runs/analysis/eqm_waypoint_exec_viz_v3_{ts}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    horizon  = int(cfg["horizon"])
    env_name = str(cfg.get("env", "maze2d-umaze-v1"))
    print(f"horizon={horizon}, env={env_name}")
    print(f"eps_wp={args.eps_wp}, waypoint_strategy={args.waypoint_strategy}")

    replay_obs = get_replay_observations(dataset)
    env = gym.make(env_name)
    rng = np.random.default_rng(args.seed)

    # Prime the sim and build wall geometry
    env.reset()
    env.sim.forward()
    obs_offset = get_particle_offset(env)
    wall_rects = build_wall_rects(env, obs_offset)

    # Diagnostic: coordinate alignment check
    obs_xy = replay_obs[:, :2]
    print(f"Replay obs xy range: x=[{obs_xy[:,0].min():.2f},{obs_xy[:,0].max():.2f}]"
          f"  y=[{obs_xy[:,1].min():.2f},{obs_xy[:,1].max():.2f}]")
    print(f"Particle body offset (obs→world): {obs_offset}")
    if wall_rects:
        wxs = [r[0] for r in wall_rects]
        wys = [r[1] for r in wall_rects]
        print(f"Wall obs-frame range: x=[{min(wxs):.2f},{max(wxs):.2f}]"
              f"  y=[{min(wys):.2f},{max(wys):.2f}]"
              f"  ({len(wall_rects)} wall geoms)")

    # Pre-filter replay positions that satisfy wall distance constraint
    valid_idx = filter_replay_by_wall_dist(replay_obs, wall_rects, args.min_wall_dist)
    n_total = len(replay_obs)
    n_valid = len(valid_idx)
    print(f"Wall-distance filter (min_wall_dist={args.min_wall_dist}): "
          f"{n_valid}/{n_total} replay positions valid ({n_valid/n_total:.1%})")

    # Run episodes
    episodes = []
    for ep in range(args.n_episodes):
        start_full, goal_full = sample_start_goal_from_replay(
            replay_obs, rng,
            min_dist=args.min_start_goal_dist,
            wall_rects=wall_rects,
            min_wall_dist=args.min_wall_dist,
        )
        goal_xy  = goal_full[:2].copy()
        goal_obs = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0], dtype=np.float32)
        start_xy = start_full[:2].copy()

        if args.waypoint_strategy == "on_path":
            wp_xy = sample_on_path_waypoint(
                replay_obs, start_xy, goal_xy, rng,
                wall_rects=wall_rects,
                min_wall_dist=args.min_wall_dist,
            )
        else:  # "feasible"
            wp_xy = sample_eval_waypoint(
                mode="feasible", replay_observations=replay_obs,
                start_xy=start_xy, goal_xy=goal_xy,
                waypoint_eps=0.5, rng=rng,
            )

        s_wd = min_wall_surface_dist(start_xy, wall_rects)
        g_wd = min_wall_surface_dist(goal_xy, wall_rects)
        w_wd = min_wall_surface_dist(wp_xy, wall_rects)
        print(f"  Ep {ep+1}/{args.n_episodes}: "
              f"s→g={np.linalg.norm(start_xy-goal_xy):.2f}  "
              f"wall_d(s={s_wd:.2f},g={g_wd:.2f},wp={w_wd:.2f})",
              end=" ", flush=True)

        ep_data = run_episode_with_trajectory(
            model, dataset, env, device, horizon,
            goal_xy=goal_xy, goal_obs=goal_obs, waypoint_xy=wp_xy,
            replan_every=args.replan_every, max_steps=args.max_steps,
            eps_wp=args.eps_wp,
        )
        status = "WP-HIT" if ep_data["hit_wp"] else "WP-MISS"
        print(f"→ {status} (min_wp={ep_data['min_wp_dist']:.3f})")
        episodes.append(ep_data)

    # Grid plot
    n    = len(episodes)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for i, ep_data in enumerate(episodes):
        r, c = divmod(i, cols)
        plot_episode(ep_data, axes[r, c], env,
                     title=f"Episode {i+1}", eps_wp=args.eps_wp,
                     obs_offset=obs_offset)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    axes[0, 0].legend(fontsize=7, loc="upper left")

    strat_label = (f"{args.waypoint_strategy}, eps={args.eps_wp}, "
                   f"min_wall={args.min_wall_dist}")
    fig.suptitle(f"H3 Exec-Level Waypoint Trajectories ({strat_label})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    plot_path = os.path.join(outdir, "waypoint_exec_trajectories_v3.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {plot_path}")

    n_hit  = sum(1 for e in episodes if e["hit_wp"])
    n_goal = sum(1 for e in episodes if e["hit_goal"])
    print(f"\n{'='*60}")
    print(f"Summary: {n} episodes, eps_wp={args.eps_wp}, "
          f"min_wall_dist={args.min_wall_dist}")
    print(f"WP hit:  {n_hit}/{n}  ({n_hit/n:.1%})")
    print(f"Goal hit: {n_goal}/{n} ({n_goal/n:.1%})")
    print(f"Mean min WP dist: {np.mean([e['min_wp_dist'] for e in episodes]):.3f}")
    print(f"Mean min goal dist: {np.mean([e['min_goal_dist'] for e in episodes]):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
