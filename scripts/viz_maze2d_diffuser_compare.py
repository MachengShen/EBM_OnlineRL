#!/usr/bin/env python3
"""Compare EqM vs DDPM-Diffuser imagined + executed trajectories on Maze2D.

For the same set of start/goal pairs shows side-by-side:
  Left panel:  EqM imagined trajectory (dashed green) + executed path (blue)
  Right panel: Diffuser imagined trajectory (dashed purple) + executed path (blue)

The "imagined" trajectory is the full planned path the model produces before
any physics execution.  If it goes through walls the model lacks wall-avoidance;
if it routes around them the model has learned maze navigation.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \\
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \\
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \\
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \\
    scripts/viz_maze2d_diffuser_compare.py \\
    --eqm_checkpoint <path> \\
    --diffuser_checkpoint <path> \\
    --n_episodes 8
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
    build_wall_rects,
    draw_maze_walls,
    filter_replay_by_wall_dist,
    get_particle_offset,
    get_replay_observations,
    load_diffuser_model_and_dataset,
    load_eqm_model_and_dataset,
    min_wall_surface_dist,
    normalize_condition,
    sample_start_goal_from_replay,
)
from synthetic_maze2d_diffuser_probe import safe_reset, safe_step

import d4rl  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory planning helpers
# ─────────────────────────────────────────────────────────────────────────────

def plan_imagined_eqm(eqm_model, eqm_dataset, start_obs, goal_obs, device, horizon):
    """Return imagined observation trajectory [H, OBS_DIM] from EqM model."""
    start_norm = normalize_condition(eqm_dataset, start_obs, device)
    goal_norm  = normalize_condition(eqm_dataset, goal_obs, device)
    cond = {0: start_norm, horizon - 1: goal_norm}
    with torch.no_grad():
        traj = eqm_model.conditional_sample(cond, horizon=horizon, verbose=False)
    # traj: (1, H, transition_dim) — tensor
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    return eqm_dataset.normalizer.unnormalize(obs_norm, "observations")[0]  # (H, OBS_DIM)


def plan_imagined_diffuser(diff_model, diff_dataset, start_obs, goal_obs, device, horizon):
    """Return imagined observation trajectory [H, OBS_DIM] from Diffuser model.

    GaussianDiffusion.conditional_sample() in this codebase (diffuser-maze2d)
    returns the trajectory tensor directly with shape (B, H, transition_dim).
    """
    start_norm = normalize_condition(diff_dataset, start_obs, device)
    goal_norm  = normalize_condition(diff_dataset, goal_obs, device)
    cond = {0: start_norm, horizon - 1: goal_norm}
    with torch.no_grad():
        traj = diff_model.conditional_sample(cond, horizon=horizon, verbose=False)
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    return diff_dataset.normalizer.unnormalize(obs_norm, "observations")[0]  # (H, OBS_DIM)


def plan_actions_eqm(eqm_model, eqm_dataset, start_obs, goal_obs, device, horizon):
    """Return (actions, observations) for EqM plan."""
    start_norm = normalize_condition(eqm_dataset, start_obs, device)
    goal_norm  = normalize_condition(eqm_dataset, goal_obs, device)
    cond = {0: start_norm, horizon - 1: goal_norm}
    with torch.no_grad():
        traj = eqm_model.conditional_sample(cond, horizon=horizon, verbose=False)
    act_norm = traj[:, :, :ACT_DIM].detach().cpu().numpy()
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    acts = eqm_dataset.normalizer.unnormalize(act_norm, "actions")[0]
    obs  = eqm_dataset.normalizer.unnormalize(obs_norm, "observations")[0]
    return acts, obs


def plan_actions_diffuser(diff_model, diff_dataset, start_obs, goal_obs, device, horizon):
    """Return (actions, observations) for Diffuser plan."""
    start_norm = normalize_condition(diff_dataset, start_obs, device)
    goal_norm  = normalize_condition(diff_dataset, goal_obs, device)
    cond = {0: start_norm, horizon - 1: goal_norm}
    with torch.no_grad():
        traj = diff_model.conditional_sample(cond, horizon=horizon, verbose=False)
    act_norm = traj[:, :, :ACT_DIM].detach().cpu().numpy()
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    acts = diff_dataset.normalizer.unnormalize(act_norm, "actions")[0]
    obs  = diff_dataset.normalizer.unnormalize(obs_norm, "observations")[0]
    return acts, obs


# ─────────────────────────────────────────────────────────────────────────────
# Execution helper (generic, works for both models via plan_fn)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(plan_fn, env, goal_xy, goal_obs, device,
                replan_every=8, max_steps=300, eps_goal=0.5):
    """Execute one episode using plan_fn for replanning.

    plan_fn(start_obs, goal_obs) -> (actions, observations)

    Returns dict with executed trajectory, first imagined path, and hit flag.
    """
    obs = safe_reset(env)
    act_low  = env.action_space.low
    act_high = env.action_space.high

    trajectory = [obs[:2].copy()]
    first_imagined = None  # first imagined path (planned observations)
    planned_actions = None
    plan_offset = 0
    min_goal_dist = float("inf")
    hit_goal = False

    for step in range(max_steps):
        need_replan = (
            planned_actions is None
            or step % replan_every == 0
            or plan_offset >= len(planned_actions)
        )
        if need_replan:
            pa, po = plan_fn(obs, goal_obs)
            planned_actions = pa
            plan_offset = 0
            if first_imagined is None:
                first_imagined = po[:, :2].copy()  # (H, 2) xy only

        action = planned_actions[plan_offset] if plan_offset < len(planned_actions) \
                 else np.zeros(ACT_DIM, dtype=np.float32)
        plan_offset += 1
        action = np.clip(action, act_low, act_high).astype(np.float32)
        obs, _reward, done, _info = safe_step(env, action)
        trajectory.append(obs[:2].copy())

        dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = min(min_goal_dist, dist)
        if dist <= eps_goal:
            hit_goal = True

        if done:
            break

    return {
        "trajectory":    np.array(trajectory),
        "first_imagined": first_imagined,
        "min_goal_dist":  min_goal_dist,
        "hit_goal":       hit_goal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_panel(ax, ep_result, start_xy, goal_xy, env,
                          obs_offset, title="", model_label="", eps_goal=0.5,
                          plan_color="#00bb55", executed=True):
    """Plot one panel: imagined plan + executed trajectory + maze."""
    draw_maze_walls(ax, env, obs_offset=obs_offset)

    # Imagined trajectory
    imag = ep_result["first_imagined"]
    if imag is not None:
        ax.plot(imag[:, 0], imag[:, 1], "--",
                color=plan_color, alpha=0.75, linewidth=2.0,
                label=f"{model_label} imagined", zorder=2)

    # Executed trajectory
    if executed:
        traj = ep_result["trajectory"]
        ax.plot(traj[:, 0], traj[:, 1], "-",
                color="#2255cc", alpha=0.8, linewidth=1.8,
                label="executed", zorder=3)

    # Start / goal markers
    ax.plot(start_xy[0], start_xy[1],
            "o", color="black", markersize=9, zorder=5, label="start")
    ax.plot(goal_xy[0], goal_xy[1],
            "*", color="red",   markersize=14, zorder=5, label="goal")

    goal_str = "HIT" if ep_result["hit_goal"] else f"MISS ({ep_result['min_goal_dist']:.2f})"
    ax.set_title(f"{title}\nGoal: {goal_str}", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqm_checkpoint",      required=True,
                        help="Path to EqM checkpoint_last.pt")
    parser.add_argument("--diffuser_checkpoint", required=True,
                        help="Path to Diffuser checkpoint_last.pt")
    parser.add_argument("--n_episodes",      type=int,   default=8)
    parser.add_argument("--replan_every",    type=int,   default=8)
    parser.add_argument("--max_steps",       type=int,   default=300)
    parser.add_argument("--eps_goal",        type=float, default=0.5)
    parser.add_argument("--min_wall_dist",   type=float, default=0.3)
    parser.add_argument("--min_start_goal_dist", type=float, default=1.5)
    parser.add_argument("--no_execute",      action="store_true",
                        help="Skip execution; only show imagined trajectories")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or f"runs/analysis/diffuser_compare_{ts}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading EqM checkpoint:      {args.eqm_checkpoint}")
    eqm_model, eqm_dataset, eqm_cfg = load_eqm_model_and_dataset(
        args.eqm_checkpoint, device)
    eqm_horizon  = int(eqm_cfg["horizon"])
    env_name     = str(eqm_cfg.get("env", "maze2d-umaze-v1"))

    print(f"Loading Diffuser checkpoint: {args.diffuser_checkpoint}")
    diff_model, diff_dataset, diff_cfg = load_diffuser_model_and_dataset(
        args.diffuser_checkpoint, device)
    diff_horizon = int(diff_cfg["horizon"])

    print(f"EqM horizon={eqm_horizon}, Diffuser horizon={diff_horizon}, env={env_name}")

    replay_obs = get_replay_observations(eqm_dataset)
    env = gym.make(env_name)
    rng = np.random.default_rng(args.seed)

    env.reset()
    env.sim.forward()
    obs_offset = get_particle_offset(env)
    wall_rects = build_wall_rects(env, obs_offset)

    obs_xy = replay_obs[:, :2]
    valid_idx = filter_replay_by_wall_dist(replay_obs, wall_rects, args.min_wall_dist)
    print(f"Particle offset: {obs_offset}")
    print(f"Replay xy: x=[{obs_xy[:,0].min():.2f},{obs_xy[:,0].max():.2f}] "
          f"y=[{obs_xy[:,1].min():.2f},{obs_xy[:,1].max():.2f}]")
    print(f"Wall-distance filter ({args.min_wall_dist}): "
          f"{len(valid_idx)}/{len(replay_obs)} valid ({len(valid_idx)/len(replay_obs):.1%})")

    # Build plan functions for each model
    def eqm_plan_fn(start_obs, goal_obs):
        return plan_actions_eqm(eqm_model, eqm_dataset, start_obs, goal_obs,
                                device, eqm_horizon)

    def diff_plan_fn(start_obs, goal_obs):
        return plan_actions_diffuser(diff_model, diff_dataset, start_obs, goal_obs,
                                     device, diff_horizon)

    # Collect episodes
    n = args.n_episodes
    episodes = []  # list of dicts with eqm/diffuser results + metadata

    for ep in range(n):
        start_full, goal_full = sample_start_goal_from_replay(
            replay_obs, rng,
            min_dist=args.min_start_goal_dist,
            wall_rects=wall_rects,
            min_wall_dist=args.min_wall_dist,
        )
        start_xy  = start_full[:2].copy()
        goal_xy   = goal_full[:2].copy()
        goal_obs  = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0], dtype=np.float32)
        start_obs = start_full.copy()

        s_wd = min_wall_surface_dist(start_xy, wall_rects)
        g_wd = min_wall_surface_dist(goal_xy, wall_rects)
        dist = float(np.linalg.norm(start_xy - goal_xy))
        print(f"  Ep {ep+1}/{n}: s→g={dist:.2f}  wall_d(s={s_wd:.2f},g={g_wd:.2f})",
              end="", flush=True)

        # EqM: get imagined trajectory (always) + optional execution
        if not args.no_execute:
            eqm_result = run_episode(
                lambda so, go: eqm_plan_fn(so, go),
                env, goal_xy, goal_obs, device,
                replan_every=args.replan_every,
                max_steps=args.max_steps,
                eps_goal=args.eps_goal,
            )
        else:
            imag_obs = plan_imagined_eqm(eqm_model, eqm_dataset, start_obs, goal_obs,
                                          device, eqm_horizon)
            eqm_result = {
                "trajectory":    None,
                "first_imagined": imag_obs[:, :2],
                "min_goal_dist":  float("nan"),
                "hit_goal":       False,
            }

        # Diffuser: get imagined trajectory (always) + optional execution
        if not args.no_execute:
            diff_result = run_episode(
                lambda so, go: diff_plan_fn(so, go),
                env, goal_xy, goal_obs, device,
                replan_every=args.replan_every,
                max_steps=args.max_steps,
                eps_goal=args.eps_goal,
            )
        else:
            imag_obs = plan_imagined_diffuser(diff_model, diff_dataset, start_obs, goal_obs,
                                               device, diff_horizon)
            diff_result = {
                "trajectory":    None,
                "first_imagined": imag_obs[:, :2],
                "min_goal_dist":  float("nan"),
                "hit_goal":       False,
            }

        eqm_g  = "HIT" if eqm_result["hit_goal"]  else f"MISS"
        diff_g = "HIT" if diff_result["hit_goal"] else f"MISS"
        print(f"  EqM:{eqm_g}  Diff:{diff_g}")

        episodes.append({
            "start_xy": start_xy,
            "goal_xy":  goal_xy,
            "eqm":  eqm_result,
            "diff": diff_result,
        })

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Layout: pairs_per_row episode pairs, each pair = 2 adjacent columns (EqM | Diff)
    pairs_per_row = 2  # how many episode pairs to show per row
    n_cols = pairs_per_row * 2  # EqM col + Diff col per pair
    n_rows = (n + pairs_per_row - 1) // pairs_per_row

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    for ep_idx, ep in enumerate(episodes):
        pair_in_row = ep_idx % pairs_per_row
        row         = ep_idx // pairs_per_row
        col_eqm     = pair_in_row * 2
        col_diff    = pair_in_row * 2 + 1

        eqm_ax  = axes[row, col_eqm]
        diff_ax = axes[row, col_diff]

        executed = not args.no_execute

        plot_comparison_panel(
            eqm_ax, ep["eqm"],
            ep["start_xy"], ep["goal_xy"], env, obs_offset,
            title=f"Ep {ep_idx+1} — EqM (h={eqm_horizon})",
            model_label="EqM",
            plan_color="#00bb55",
            executed=executed,
        )
        plot_comparison_panel(
            diff_ax, ep["diff"],
            ep["start_xy"], ep["goal_xy"], env, obs_offset,
            title=f"Ep {ep_idx+1} — Diffuser (h={diff_horizon})",
            model_label="Diffuser",
            plan_color="#9933cc",
            executed=executed,
        )

    # Hide unused axes
    for idx in range(n, n_rows * pairs_per_row):
        row = idx // pairs_per_row
        col = (idx % pairs_per_row) * 2
        if row < n_rows and col < n_cols:
            axes[row, col].set_visible(False)
            if col + 1 < n_cols:
                axes[row, col + 1].set_visible(False)

    # Add a shared legend on the first axes
    handles = [
        mpatches.Patch(color="#00bb55", label="EqM imagined (dashed)"),
        mpatches.Patch(color="#9933cc", label="Diffuser imagined (dashed)"),
        mpatches.Patch(color="#2255cc", label="Executed path"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="upper left")

    mode_label = "imagined only" if args.no_execute else "imagined + executed"
    fig.suptitle(
        f"EqM vs Diffuser — Maze2D Trajectory Comparison ({mode_label})\n"
        f"EqM h={eqm_horizon}  |  Diffuser h={diff_horizon}  |  "
        f"min_wall_dist={args.min_wall_dist}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    plot_path = os.path.join(outdir, "diffuser_compare.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {plot_path}")

    if not args.no_execute:
        eqm_hits  = sum(1 for ep in episodes if ep["eqm"]["hit_goal"])
        diff_hits = sum(1 for ep in episodes if ep["diff"]["hit_goal"])
        print(f"\nGoal hit — EqM: {eqm_hits}/{n} ({eqm_hits/n:.1%})  "
              f"Diffuser: {diff_hits}/{n} ({diff_hits/n:.1%})")


if __name__ == "__main__":
    main()
