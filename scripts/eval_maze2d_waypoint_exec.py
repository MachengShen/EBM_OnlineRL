#!/usr/bin/env python3
"""H3 follow-up: execution-level waypoint eval with env rollout + replanning.

Tests whether EqM with waypoint constraints produces plans that, when
executed step-by-step in the real environment with replanning, actually
cause the agent to visit the waypoint and reach the goal.

Uses a "shifting waypoint index" so the semantic meaning of the waypoint
is preserved as planning progresses:
  t_wp_local = t_wp_global - k_exec
  Include waypoint if 1 <= t_wp_local <= H-2, else drop it.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/eval_maze2d_waypoint_exec.py \
    --checkpoint <path> --waypoint_mode pos_only --n_episodes 50
"""
from __future__ import annotations

import argparse
import json
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
    TRANSITION_DIM,
    eqm_sample_with_pos_only_waypoint,
    get_replay_observations,
    load_eqm_model_and_dataset,
    normalize_condition,
    sample_eval_waypoint,
)
from synthetic_maze2d_diffuser_probe import safe_reset, safe_step

import d4rl  # noqa: F401


def plan_from_obs(model, dataset, obs, goal_obs, device, horizon,
                  waypoint_xy=None, waypoint_t=None,
                  waypoint_mode="pos_only", n_samples=1):
    """Plan a trajectory from current observation to goal.

    Returns:
        best_actions [H, act_dim] (unnormalized)
        best_obs [H, obs_dim] (unnormalized)
    """
    start_norm = normalize_condition(dataset, obs, device).repeat(n_samples, 1)
    goal_norm = normalize_condition(dataset, goal_obs, device).repeat(n_samples, 1)
    cond = {0: start_norm, horizon - 1: goal_norm}

    use_wp = (waypoint_xy is not None and waypoint_t is not None
              and 1 <= waypoint_t <= horizon - 2)

    if use_wp and waypoint_mode == "pos_and_zero_vel":
        wp_obs = np.zeros(OBS_DIM, dtype=np.float32)
        wp_obs[:2] = waypoint_xy
        wp_norm = normalize_condition(dataset, wp_obs, device).repeat(n_samples, 1)
        cond[waypoint_t] = wp_norm
        traj = model.conditional_sample(cond, horizon=horizon, verbose=False)
    elif use_wp and waypoint_mode == "pos_only":
        wp_obs = np.zeros(OBS_DIM, dtype=np.float32)
        wp_obs[:2] = waypoint_xy
        wp_full_norm = normalize_condition(dataset, wp_obs, device)
        wp_xy_norm = wp_full_norm[0, :2].repeat(n_samples, 1)
        traj = eqm_sample_with_pos_only_waypoint(
            model, cond, waypoint_t, wp_xy_norm, horizon)
    else:
        traj = model.conditional_sample(cond, horizon=horizon, verbose=False)

    # Extract and unnormalize
    act_norm = traj[:, :, :ACT_DIM].detach().cpu().numpy()
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    actions = dataset.normalizer.unnormalize(act_norm, "actions")
    observations = dataset.normalizer.unnormalize(obs_norm, "observations")

    if n_samples == 1:
        return actions[0], observations[0]

    # Pick trajectory closest to goal at the end
    goal_xy = goal_obs[:2]
    scores = [np.linalg.norm(observations[b, -1, :2] - goal_xy)
              for b in range(n_samples)]
    best_idx = int(np.argmin(scores))
    return actions[best_idx], observations[best_idx]


def run_episode(model, dataset, env, device, horizon, goal_xy, goal_obs,
                replay_obs, rng,
                waypoint_xy=None, waypoint_mode="pos_only",
                replan_every=1, max_steps=300,
                eps_wp=0.5, eps_goal=0.5, n_plan_samples=1):
    """Run a single episode with replanning and shifting waypoint index.

    Returns dict of per-episode metrics.
    """
    obs = safe_reset(env)

    t_wp_global = (horizon - 1) // 2
    k_exec = 0
    planned_actions = None
    plan_offset = 0
    n_replans = 0

    hit_waypoint = False
    hit_goal = False
    steps_to_wp = None
    steps_to_goal = None
    min_wp_dist = float("inf")
    min_goal_dist = float("inf")

    act_low = env.action_space.low
    act_high = env.action_space.high

    for step in range(max_steps):
        need_replan = (planned_actions is None
                       or step % replan_every == 0
                       or plan_offset >= len(planned_actions))

        if need_replan:
            # Shifting waypoint index
            t_wp_local = t_wp_global - k_exec
            use_wp = (waypoint_xy is not None
                      and 1 <= t_wp_local <= horizon - 2)

            with torch.no_grad():
                planned_actions, _ = plan_from_obs(
                    model, dataset, obs, goal_obs, device, horizon,
                    waypoint_xy=waypoint_xy if use_wp else None,
                    waypoint_t=t_wp_local if use_wp else None,
                    waypoint_mode=waypoint_mode,
                    n_samples=n_plan_samples,
                )
            plan_offset = 0
            n_replans += 1

        # Execute action
        if plan_offset < len(planned_actions):
            action = planned_actions[plan_offset]
            plan_offset += 1
        else:
            action = np.zeros(ACT_DIM, dtype=np.float32)

        action = np.clip(action, act_low, act_high).astype(np.float32)
        obs, reward, done, info = safe_step(env, action)
        k_exec += 1

        # Track waypoint proximity
        if waypoint_xy is not None:
            wp_dist = float(np.linalg.norm(obs[:2] - waypoint_xy))
            min_wp_dist = min(min_wp_dist, wp_dist)
            if wp_dist <= eps_wp and not hit_waypoint:
                hit_waypoint = True
                steps_to_wp = step + 1

        # Track goal proximity
        goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = min(min_goal_dist, goal_dist)
        if goal_dist <= eps_goal and not hit_goal:
            hit_goal = True
            steps_to_goal = step + 1

        if done:
            break

    return {
        "hit_waypoint": hit_waypoint,
        "hit_goal": hit_goal,
        "joint_success": hit_waypoint and hit_goal,
        "steps_to_wp": steps_to_wp,
        "steps_to_goal": steps_to_goal,
        "min_wp_dist": min_wp_dist,
        "min_goal_dist": min_goal_dist,
        "n_replans": n_replans,
        "total_steps": k_exec,
    }


def sample_start_goal_from_replay(replay_obs, rng, min_dist=1.0, max_attempts=500):
    """Sample start/goal pair with minimum xy distance from replay."""
    xy = replay_obs[:, :2]
    for _ in range(max_attempts):
        idx_s = rng.integers(len(xy))
        idx_g = rng.integers(len(xy))
        if np.linalg.norm(xy[idx_s] - xy[idx_g]) >= min_dist:
            return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()
    # Fallback
    idx_s = rng.integers(len(xy))
    dists = np.linalg.norm(xy - xy[idx_s][None, :], axis=1)
    idx_g = int(np.argmax(dists))
    return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()


def run_eval(model, dataset, env, device, horizon, replay_obs,
             waypoint_mode, n_episodes, replan_every, max_steps,
             eps_wp, eps_goal, n_plan_samples, seed, use_waypoint=True):
    """Run full evaluation across episodes.

    Returns dict of aggregated metrics + per-episode details.
    """
    rng = np.random.default_rng(seed)
    results = []

    for ep in range(n_episodes):
        # Sample start and goal from replay
        start_full, goal_full = sample_start_goal_from_replay(replay_obs, rng)
        start_xy = start_full[:2].copy()
        goal_xy = goal_full[:2].copy()
        goal_obs = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0], dtype=np.float32)

        # Sample waypoint
        wp_xy = None
        if use_waypoint:
            wp_xy = sample_eval_waypoint(
                mode="feasible",
                replay_observations=replay_obs,
                start_xy=start_xy,
                goal_xy=goal_xy,
                waypoint_eps=eps_wp,
                rng=rng,
            )
            if wp_xy is None:
                # Fallback: random replay position not too close to goal
                for _ in range(100):
                    idx = rng.integers(len(replay_obs))
                    candidate = replay_obs[idx, :2].copy()
                    if np.linalg.norm(candidate - goal_xy) > eps_wp:
                        wp_xy = candidate
                        break

        ep_result = run_episode(
            model, dataset, env, device, horizon,
            goal_xy=goal_xy, goal_obs=goal_obs,
            replay_obs=replay_obs, rng=rng,
            waypoint_xy=wp_xy, waypoint_mode=waypoint_mode,
            replan_every=replan_every, max_steps=max_steps,
            eps_wp=eps_wp, eps_goal=eps_goal,
            n_plan_samples=n_plan_samples,
        )
        ep_result["episode"] = ep
        ep_result["goal_xy"] = goal_xy.tolist()
        ep_result["waypoint_xy"] = wp_xy.tolist() if wp_xy is not None else None
        results.append(ep_result)

        if (ep + 1) % 10 == 0 or ep == 0:
            wp_rate = sum(r["hit_waypoint"] for r in results) / len(results)
            goal_rate = sum(r["hit_goal"] for r in results) / len(results)
            joint_rate = sum(r["joint_success"] for r in results) / len(results)
            print(f"  [{ep+1}/{n_episodes}] wp={wp_rate:.3f} goal={goal_rate:.3f} "
                  f"joint={joint_rate:.3f}")
            sys.stdout.flush()

    # Aggregate
    n = len(results)
    agg = {
        "waypoint_hit_rate": sum(r["hit_waypoint"] for r in results) / max(1, n),
        "goal_hit_rate": sum(r["hit_goal"] for r in results) / max(1, n),
        "joint_success_rate": sum(r["joint_success"] for r in results) / max(1, n),
        "mean_min_wp_dist": float(np.mean([r["min_wp_dist"] for r in results])),
        "mean_min_goal_dist": float(np.mean([r["min_goal_dist"] for r in results])),
        "mean_steps_to_wp": float(np.mean([
            r["steps_to_wp"] for r in results if r["steps_to_wp"] is not None
        ])) if any(r["steps_to_wp"] is not None for r in results) else None,
        "mean_steps_to_goal": float(np.mean([
            r["steps_to_goal"] for r in results if r["steps_to_goal"] is not None
        ])) if any(r["steps_to_goal"] is not None for r in results) else None,
        "mean_replans": float(np.mean([r["n_replans"] for r in results])),
        "n_episodes": n,
        "use_waypoint": use_waypoint,
        "waypoint_mode": waypoint_mode if use_waypoint else "none",
    }
    return agg, results


def main():
    parser = argparse.ArgumentParser(description="H3 execution-level waypoint eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--waypoint_mode", default="pos_only",
                        choices=["pos_only", "pos_and_zero_vel", "both"])
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--replan_every", type=int, default=1,
                        help="Replan every N env steps (1=every step)")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--eps_wp", type=float, default=0.5)
    parser.add_argument("--eps_goal", type=float, default=0.5)
    parser.add_argument("--n_plan_samples", type=int, default=1)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or f"runs/analysis/eqm_waypoint_exec_h3_{ts}"
    os.makedirs(outdir, exist_ok=True)

    print(f"[H3 exec] Loading checkpoint: {args.checkpoint}")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    horizon = int(cfg["horizon"])
    env_name = str(cfg.get("env", "maze2d-umaze-v1"))
    print(f"[H3 exec] horizon={horizon}, env={env_name}, "
          f"replan_every={args.replan_every}, max_steps={args.max_steps}")

    replay_obs = get_replay_observations(dataset)
    print(f"[H3 exec] Replay observations: {replay_obs.shape}")

    env = gym.make(env_name)

    modes = (["pos_only", "pos_and_zero_vel"]
             if args.waypoint_mode == "both" else [args.waypoint_mode])

    all_results = {}

    for mode in modes:
        print(f"\n=== Mode: {mode} (with waypoint) ===")
        sys.stdout.flush()
        agg, details = run_eval(
            model, dataset, env, device, horizon, replay_obs,
            waypoint_mode=mode, n_episodes=args.n_episodes,
            replan_every=args.replan_every, max_steps=args.max_steps,
            eps_wp=args.eps_wp, eps_goal=args.eps_goal,
            n_plan_samples=args.n_plan_samples, seed=args.seed,
            use_waypoint=True,
        )
        all_results[mode] = agg
        print(f"  {mode}: wp={agg['waypoint_hit_rate']:.3f} "
              f"goal={agg['goal_hit_rate']:.3f} joint={agg['joint_success_rate']:.3f}")

        # Save per-mode details
        detail_path = os.path.join(outdir, f"exec_details_{mode}.json")
        with open(detail_path, "w") as f:
            json.dump(details, f, indent=2, default=str)

    # Baseline: no waypoint
    print(f"\n=== Baseline: no waypoint ===")
    sys.stdout.flush()
    agg_base, details_base = run_eval(
        model, dataset, env, device, horizon, replay_obs,
        waypoint_mode="pos_only", n_episodes=args.n_episodes,
        replan_every=args.replan_every, max_steps=args.max_steps,
        eps_wp=args.eps_wp, eps_goal=args.eps_goal,
        n_plan_samples=args.n_plan_samples, seed=args.seed,
        use_waypoint=False,
    )
    all_results["no_waypoint"] = agg_base
    print(f"  no_waypoint: wp={agg_base['waypoint_hit_rate']:.3f} "
          f"goal={agg_base['goal_hit_rate']:.3f} joint={agg_base['joint_success_rate']:.3f}")

    detail_path = os.path.join(outdir, "exec_details_no_waypoint.json")
    with open(detail_path, "w") as f:
        json.dump(details_base, f, indent=2, default=str)

    # Save combined summary
    summary = {
        "hypothesis": "H3_exec",
        "description": "Execution-level waypoint eval with env rollout + replanning",
        "checkpoint": str(args.checkpoint),
        "env": env_name,
        "horizon": horizon,
        "replan_every": args.replan_every,
        "max_steps": args.max_steps,
        "eps_wp": args.eps_wp,
        "eps_goal": args.eps_goal,
        "n_plan_samples": args.n_plan_samples,
        "n_episodes": args.n_episodes,
        "modes": all_results,
    }
    with open(os.path.join(outdir, "exec_waypoint_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== H3 Execution-Level Waypoint Results ===")
    for mode, agg in all_results.items():
        print(f"  {mode:20s}: wp={agg['waypoint_hit_rate']:.3f} "
              f"goal={agg['goal_hit_rate']:.3f} joint={agg['joint_success_rate']:.3f} "
              f"mean_min_wp={agg['mean_min_wp_dist']:.3f} "
              f"mean_min_goal={agg['mean_min_goal_dist']:.3f}")
    print(f"\nSaved to: {outdir}")


if __name__ == "__main__":
    main()
