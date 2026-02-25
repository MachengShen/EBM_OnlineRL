#!/usr/bin/env python3
"""H3 waypoint constraint compositionality evaluation on Maze2D.

Tests whether EqM can satisfy test-time waypoint constraints without retraining.
Implements both pos_only (position dims only) and pos_and_zero_vel (full obs) modes.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/eval_maze2d_waypoint.py --checkpoint <path> --waypoint_mode pos_only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from diffuser.models.helpers import apply_conditioning

from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    EquilibriumMatchingDiffusion,
    eqm_sample_with_pos_only_waypoint,
    get_replay_observations,
    load_eqm_model_and_dataset,
    normalize_condition,
    resolve_waypoint_t,
    sample_eval_waypoint,
)


def sample_start_goal_from_replay(
    replay_obs: np.ndarray,
    rng: np.random.Generator,
    min_dist: float = 1.0,
    max_attempts: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample start/goal pair with minimum xy distance from replay."""
    xy = replay_obs[:, :2]
    for _ in range(max_attempts):
        idx_s = rng.integers(len(xy))
        idx_g = rng.integers(len(xy))
        if np.linalg.norm(xy[idx_s] - xy[idx_g]) >= min_dist:
            return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()
    # Fallback: just use max-distance pair
    idx_s = rng.integers(len(xy))
    dists = np.linalg.norm(xy - xy[idx_s][None, :], axis=1)
    idx_g = int(np.argmax(dists))
    return replay_obs[idx_s].copy(), replay_obs[idx_g].copy()


def run_waypoint_eval(
    model: EquilibriumMatchingDiffusion,
    dataset,
    replay_obs: np.ndarray,
    mode: str,
    n_trials: int,
    horizon: int,
    waypoint_t: int,
    waypoint_eps: float,
    goal_eps: float,
    device: torch.device,
    seed: int = 0,
) -> Dict:
    """Run waypoint evaluation in specified mode.

    Returns dict with metrics.
    """
    rng = np.random.default_rng(seed)
    n_plan_samples = 4  # plan multiple trajectories, pick best

    results = {
        "waypoint_hits": 0,
        "goal_hits": 0,
        "joint_successes": 0,
        "n_trials": n_trials,
    }
    trial_details = []

    for trial in range(n_trials):
        # Sample start, goal from replay
        start_obs, goal_obs = sample_start_goal_from_replay(replay_obs, rng)
        start_xy = start_obs[:2]
        goal_xy = goal_obs[:2]

        # Sample feasible waypoint from replay
        wp_xy = sample_eval_waypoint(
            mode="feasible",
            replay_observations=replay_obs,
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoint_eps=waypoint_eps,
            rng=rng,
        )
        if wp_xy is None:
            continue

        # Build conditioning
        start_norm = normalize_condition(dataset, start_obs, device).repeat(n_plan_samples, 1)
        goal_norm = normalize_condition(dataset, goal_obs, device).repeat(n_plan_samples, 1)
        cond = {
            0: start_norm,
            horizon - 1: goal_norm,
        }

        if mode == "pos_and_zero_vel":
            # Full obs waypoint: [x, y, 0, 0] normalized → clamps all obs dims
            wp_obs = np.zeros(OBS_DIM, dtype=np.float32)
            wp_obs[:2] = wp_xy
            wp_norm = normalize_condition(dataset, wp_obs, device).repeat(n_plan_samples, 1)
            cond[waypoint_t] = wp_norm
            # Use standard conditional_sample
            with torch.no_grad():
                traj = model.conditional_sample(cond, horizon=horizon, verbose=False)
        elif mode == "pos_only":
            # Position-only: clamp ONLY xy dims at waypoint, leave velocity free
            wp_obs_for_norm = np.zeros(OBS_DIM, dtype=np.float32)
            wp_obs_for_norm[:2] = wp_xy
            wp_full_norm = normalize_condition(dataset, wp_obs_for_norm, device)
            # Extract only the normalized xy (first 2 obs dims)
            wp_xy_norm = wp_full_norm[0, :2].repeat(n_plan_samples, 1)  # [B, 2]
            with torch.no_grad():
                traj = eqm_sample_with_pos_only_waypoint(
                    model, cond, waypoint_t, wp_xy_norm, horizon,
                )
        else:
            raise ValueError(f"Unknown waypoint_mode: {mode}")

        # Unnormalize observations for metric computation
        # Obs are at dims [act_dim:] in [act | obs] packing
        obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()  # [B, H, obs_dim]
        obs_raw = dataset.normalizer.unnormalize(obs_norm, "observations")  # [B, H, obs_dim]
        obs_xy = obs_raw[:, :, :2]  # [B, H, 2] — position only

        # Compute metrics across plan samples, take best
        best_wp_dist = float("inf")
        best_goal_dist = float("inf")
        for b in range(n_plan_samples):
            traj_xy = obs_xy[b]  # [H, 2]
            wp_dists = np.linalg.norm(traj_xy - wp_xy[None, :], axis=1)
            goal_dists = np.linalg.norm(traj_xy - goal_xy[None, :], axis=1)
            min_wp = float(np.min(wp_dists))
            min_goal = float(np.min(goal_dists))
            if min_wp + min_goal < best_wp_dist + best_goal_dist:
                best_wp_dist = min_wp
                best_goal_dist = min_goal

        wp_hit = best_wp_dist <= waypoint_eps
        goal_hit = best_goal_dist <= goal_eps
        joint = wp_hit and goal_hit

        results["waypoint_hits"] += int(wp_hit)
        results["goal_hits"] += int(goal_hit)
        results["joint_successes"] += int(joint)

        trial_details.append({
            "trial": trial,
            "waypoint_min_dist": best_wp_dist,
            "goal_min_dist": best_goal_dist,
            "waypoint_hit": wp_hit,
            "goal_hit": goal_hit,
            "joint": joint,
        })

        if (trial + 1) % 20 == 0:
            wp_rate = results["waypoint_hits"] / (trial + 1)
            goal_rate = results["goal_hits"] / (trial + 1)
            joint_rate = results["joint_successes"] / (trial + 1)
            print(f"  [{trial+1}/{n_trials}] wp_hit={wp_rate:.3f} goal_hit={goal_rate:.3f} joint={joint_rate:.3f}")

    results["waypoint_hit_rate"] = results["waypoint_hits"] / max(1, n_trials)
    results["goal_hit_rate"] = results["goal_hits"] / max(1, n_trials)
    results["joint_success_rate"] = results["joint_successes"] / max(1, n_trials)
    results["mode"] = mode
    results["waypoint_eps"] = waypoint_eps
    results["goal_eps"] = goal_eps
    results["waypoint_t"] = waypoint_t
    results["trial_details"] = trial_details
    return results


def main():
    parser = argparse.ArgumentParser(description="H3: Waypoint constraint eval on Maze2D")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--waypoint_mode", type=str, default="both",
                        choices=["pos_only", "pos_and_zero_vel", "both"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--waypoint_eps", type=float, default=0.5)
    parser.add_argument("--goal_eps", type=float, default=0.5)
    parser.add_argument("--waypoint_t", type=int, default=0, help="0 = horizon//2")
    parser.add_argument("--n_plan_samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[H3 waypoint] Loading checkpoint: {args.checkpoint}")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    horizon = int(cfg["horizon"])
    waypoint_t = resolve_waypoint_t(horizon, args.waypoint_t)
    print(f"[H3 waypoint] horizon={horizon}, waypoint_t={waypoint_t}, "
          f"mode={args.waypoint_mode}, n_trials={args.n_trials}")

    replay_obs = get_replay_observations(dataset)
    print(f"[H3 waypoint] Replay observations: {replay_obs.shape}")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path(f"runs/analysis/eqm_waypoint_h3_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    modes = ["pos_only", "pos_and_zero_vel"] if args.waypoint_mode == "both" else [args.waypoint_mode]
    all_results = {}

    # Also run no-waypoint baseline for comparison
    modes_with_baseline = modes + ["no_waypoint"]

    for mode in modes_with_baseline:
        print(f"\n--- Running mode: {mode} ---")
        if mode == "no_waypoint":
            # Baseline: no waypoint constraint, just start/goal
            # We still measure waypoint proximity (should be low/random)
            res = run_waypoint_eval(
                model, dataset, replay_obs,
                mode="pos_and_zero_vel",  # doesn't matter for no-wp baseline
                n_trials=args.n_trials,
                horizon=horizon,
                waypoint_t=waypoint_t,
                waypoint_eps=args.waypoint_eps,
                goal_eps=args.goal_eps,
                device=device,
                seed=args.seed,
            )
            # Override: re-run without actually adding the waypoint constraint
            res = run_no_waypoint_baseline(
                model, dataset, replay_obs,
                n_trials=args.n_trials,
                horizon=horizon,
                waypoint_t=waypoint_t,
                waypoint_eps=args.waypoint_eps,
                goal_eps=args.goal_eps,
                device=device,
                seed=args.seed,
            )
            res["mode"] = "no_waypoint"
        else:
            res = run_waypoint_eval(
                model, dataset, replay_obs,
                mode=mode,
                n_trials=args.n_trials,
                horizon=horizon,
                waypoint_t=waypoint_t,
                waypoint_eps=args.waypoint_eps,
                goal_eps=args.goal_eps,
                device=device,
                seed=args.seed,
            )

        all_results[mode] = {k: v for k, v in res.items() if k != "trial_details"}
        print(f"\n  {mode}: wp_hit={res['waypoint_hit_rate']:.3f} "
              f"goal_hit={res['goal_hit_rate']:.3f} joint={res['joint_success_rate']:.3f}")

        # Save per-mode details
        mode_path = outdir / f"results_{mode}.json"
        with open(mode_path, "w") as f:
            json.dump(res, f, indent=2)

    # Save combined summary
    summary = {
        "hypothesis": "H3",
        "description": "Waypoint constraint compositionality",
        "checkpoint": str(args.checkpoint),
        "horizon": horizon,
        "waypoint_t": waypoint_t,
        "waypoint_eps": args.waypoint_eps,
        "goal_eps": args.goal_eps,
        "n_trials": args.n_trials,
        "modes": all_results,
    }
    with open(outdir / "waypoint_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== H3 Waypoint Results Summary ===")
    for mode, res in all_results.items():
        print(f"  {mode:20s}: wp_hit={res['waypoint_hit_rate']:.3f} "
              f"goal_hit={res['goal_hit_rate']:.3f} joint={res['joint_success_rate']:.3f}")
    print(f"\nSaved to: {outdir}")


def run_no_waypoint_baseline(
    model, dataset, replay_obs, n_trials, horizon, waypoint_t,
    waypoint_eps, goal_eps, device, seed,
) -> Dict:
    """Run evaluation without waypoint constraint as baseline."""
    rng = np.random.default_rng(seed)
    n_plan_samples = 4

    results = {
        "waypoint_hits": 0, "goal_hits": 0, "joint_successes": 0, "n_trials": n_trials,
    }
    trial_details = []

    for trial in range(n_trials):
        start_obs, goal_obs = sample_start_goal_from_replay(replay_obs, rng)
        start_xy, goal_xy = start_obs[:2], goal_obs[:2]

        # Sample a waypoint (for measurement only, NOT used in conditioning)
        wp_xy = sample_eval_waypoint(
            mode="feasible", replay_observations=replay_obs,
            start_xy=start_xy, goal_xy=goal_xy,
            waypoint_eps=waypoint_eps, rng=rng,
        )
        if wp_xy is None:
            continue

        start_norm = normalize_condition(dataset, start_obs, device).repeat(n_plan_samples, 1)
        goal_norm = normalize_condition(dataset, goal_obs, device).repeat(n_plan_samples, 1)
        cond = {0: start_norm, horizon - 1: goal_norm}

        with torch.no_grad():
            traj = model.conditional_sample(cond, horizon=horizon, verbose=False)

        obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
        obs_raw = dataset.normalizer.unnormalize(obs_norm, "observations")
        obs_xy = obs_raw[:, :, :2]

        best_wp_dist = float("inf")
        best_goal_dist = float("inf")
        for b in range(n_plan_samples):
            traj_xy = obs_xy[b]
            wp_dists = np.linalg.norm(traj_xy - wp_xy[None, :], axis=1)
            goal_dists = np.linalg.norm(traj_xy - goal_xy[None, :], axis=1)
            min_wp, min_goal = float(np.min(wp_dists)), float(np.min(goal_dists))
            if min_wp + min_goal < best_wp_dist + best_goal_dist:
                best_wp_dist, best_goal_dist = min_wp, min_goal

        wp_hit = best_wp_dist <= waypoint_eps
        goal_hit = best_goal_dist <= goal_eps
        results["waypoint_hits"] += int(wp_hit)
        results["goal_hits"] += int(goal_hit)
        results["joint_successes"] += int(wp_hit and goal_hit)
        trial_details.append({
            "trial": trial, "waypoint_min_dist": best_wp_dist,
            "goal_min_dist": best_goal_dist,
            "waypoint_hit": wp_hit, "goal_hit": goal_hit, "joint": wp_hit and goal_hit,
        })

    results["waypoint_hit_rate"] = results["waypoint_hits"] / max(1, n_trials)
    results["goal_hit_rate"] = results["goal_hits"] / max(1, n_trials)
    results["joint_success_rate"] = results["joint_successes"] / max(1, n_trials)
    results["waypoint_eps"] = waypoint_eps
    results["goal_eps"] = goal_eps
    results["waypoint_t"] = waypoint_t
    results["trial_details"] = trial_details
    return results


if __name__ == "__main__":
    main()
