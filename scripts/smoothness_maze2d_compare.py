#!/usr/bin/env python3
"""Numerical smoothness comparison of EqM vs Diffuser imagined trajectories.

Computes step-to-step displacement statistics to determine whether
the visual discontinuity in EqM trajectories is a real model property
or a plotting artifact.

Metrics per trajectory:
  - step_disp: ||pos[t+1] - pos[t]||  for t=0..H-2
  - mean_disp, std_disp, max_disp, median_disp
  - total_path_length: sum of step displacements
  - straightness: (start-goal dist) / total_path_length  (1.0 = perfectly straight)
  - direction_change: angle between consecutive displacement vectors (in degrees)

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/smoothness_maze2d_compare.py \
    --eqm_checkpoint <path> \
    --diffuser_checkpoint <path> \
    --n_episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    build_wall_rects,
    get_particle_offset,
    get_replay_observations,
    load_diffuser_model_and_dataset,
    load_eqm_model_and_dataset,
    min_wall_surface_dist,
    normalize_condition,
    sample_start_goal_from_replay,
)

import d4rl  # noqa: F401


def compute_smoothness(xy: np.ndarray) -> dict:
    """Compute smoothness metrics for an (H, 2) xy trajectory."""
    H = len(xy)
    if H < 2:
        return {}

    # Step displacements
    disps = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # (H-1,)

    # Direction changes (angle between consecutive displacement vectors)
    dvecs = np.diff(xy, axis=0)  # (H-1, 2)
    angles = []
    for i in range(len(dvecs) - 1):
        v1, v2 = dvecs[i], dvecs[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            angles.append(0.0)
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    angles = np.array(angles)

    total_length = float(np.sum(disps))
    sg_dist = float(np.linalg.norm(xy[-1] - xy[0]))

    return {
        "mean_disp": float(np.mean(disps)),
        "std_disp": float(np.std(disps)),
        "max_disp": float(np.max(disps)),
        "median_disp": float(np.median(disps)),
        "min_disp": float(np.min(disps)),
        "total_path_length": total_length,
        "start_goal_dist": sg_dist,
        "straightness": sg_dist / total_length if total_length > 1e-8 else 0.0,
        "mean_angle_change": float(np.mean(angles)) if len(angles) > 0 else 0.0,
        "std_angle_change": float(np.std(angles)) if len(angles) > 0 else 0.0,
        "max_angle_change": float(np.max(angles)) if len(angles) > 0 else 0.0,
        "disp_cv": float(np.std(disps) / np.mean(disps)) if np.mean(disps) > 1e-8 else 0.0,
        # Raw arrays for detailed analysis
        "_disps": disps,
        "_angles": angles,
    }


def plan_imagined(model, dataset, start_obs, goal_obs, device, horizon, is_eqm=True):
    """Generate imagined trajectory [H, OBS_DIM] from either model."""
    start_norm = normalize_condition(dataset, start_obs, device)
    goal_norm  = normalize_condition(dataset, goal_obs, device)
    cond = {0: start_norm, horizon - 1: goal_norm}
    with torch.no_grad():
        traj = model.conditional_sample(cond, horizon=horizon, verbose=False)
    obs_norm = traj[:, :, ACT_DIM:].detach().cpu().numpy()
    return dataset.normalizer.unnormalize(obs_norm, "observations")[0]  # (H, OBS_DIM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqm_checkpoint", required=True)
    parser.add_argument("--diffuser_checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--min_wall_dist", type=float, default=0.3)
    parser.add_argument("--min_start_goal_dist", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print("Loading EqM checkpoint...")
    eqm_model, eqm_dataset, eqm_cfg = load_eqm_model_and_dataset(
        args.eqm_checkpoint, device)
    eqm_horizon = int(eqm_cfg["horizon"])

    print("Loading Diffuser checkpoint...")
    diff_model, diff_dataset, diff_cfg = load_diffuser_model_and_dataset(
        args.diffuser_checkpoint, device)
    diff_horizon = int(diff_cfg["horizon"])

    env_name = str(eqm_cfg.get("env", "maze2d-umaze-v1"))
    print(f"EqM horizon={eqm_horizon}, Diffuser horizon={diff_horizon}, env={env_name}")

    replay_obs = get_replay_observations(eqm_dataset)
    env = gym.make(env_name)
    rng = np.random.default_rng(args.seed)

    env.reset()
    env.sim.forward()
    obs_offset = get_particle_offset(env)
    wall_rects = build_wall_rects(env, obs_offset)

    # Collect smoothness stats
    eqm_stats = []
    diff_stats = []
    eqm_all_disps = []
    diff_all_disps = []

    print(f"\nGenerating {args.n_episodes} imagined trajectory pairs...\n")
    print(f"{'Ep':>3}  {'s→g':>5}  {'EqM mean':>9} {'EqM std':>8} {'EqM max':>8}  "
          f"{'Diff mean':>9} {'Diff std':>8} {'Diff max':>8}")
    print("-" * 85)

    for ep in range(args.n_episodes):
        start_full, goal_full = sample_start_goal_from_replay(
            replay_obs, rng,
            min_dist=args.min_start_goal_dist,
            wall_rects=wall_rects,
            min_wall_dist=args.min_wall_dist,
        )
        start_obs = start_full.copy()
        goal_obs = np.array([goal_full[0], goal_full[1], 0.0, 0.0], dtype=np.float32)

        # EqM imagined
        eqm_obs = plan_imagined(eqm_model, eqm_dataset, start_obs, goal_obs,
                                device, eqm_horizon, is_eqm=True)
        eqm_xy = eqm_obs[:, :2]
        eqm_sm = compute_smoothness(eqm_xy)
        eqm_stats.append(eqm_sm)
        eqm_all_disps.extend(eqm_sm["_disps"].tolist())

        # Diffuser imagined
        diff_obs = plan_imagined(diff_model, diff_dataset, start_obs, goal_obs,
                                 device, diff_horizon, is_eqm=False)
        diff_xy = diff_obs[:, :2]
        diff_sm = compute_smoothness(diff_xy)
        diff_stats.append(diff_sm)
        diff_all_disps.extend(diff_sm["_disps"].tolist())

        sg = float(np.linalg.norm(start_full[:2] - goal_full[:2]))
        print(f"{ep+1:3d}  {sg:5.2f}  {eqm_sm['mean_disp']:9.4f} {eqm_sm['std_disp']:8.4f} "
              f"{eqm_sm['max_disp']:8.4f}  {diff_sm['mean_disp']:9.4f} "
              f"{diff_sm['std_disp']:8.4f} {diff_sm['max_disp']:8.4f}")

    # Aggregate statistics
    def agg(stats_list, key):
        vals = [s[key] for s in stats_list]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 85)
    print("AGGREGATE SMOOTHNESS COMPARISON")
    print("=" * 85)

    metrics = [
        ("mean_disp", "Mean step displacement"),
        ("std_disp", "Std step displacement"),
        ("max_disp", "Max step displacement"),
        ("median_disp", "Median step displacement"),
        ("min_disp", "Min step displacement"),
        ("disp_cv", "Coefficient of variation (std/mean)"),
        ("total_path_length", "Total path length"),
        ("straightness", "Straightness (sg_dist/path_len)"),
        ("mean_angle_change", "Mean direction change (deg)"),
        ("max_angle_change", "Max direction change (deg)"),
    ]

    print(f"\n{'Metric':<38}  {'EqM':>16}  {'Diffuser':>16}  {'Ratio (E/D)':>12}")
    print("-" * 90)

    for key, label in metrics:
        eqm_mean, eqm_std = agg(eqm_stats, key)
        diff_mean, diff_std = agg(diff_stats, key)
        ratio = eqm_mean / diff_mean if abs(diff_mean) > 1e-8 else float("inf")
        print(f"{label:<38}  {eqm_mean:7.4f}±{eqm_std:6.4f}  "
              f"{diff_mean:7.4f}±{diff_std:6.4f}  {ratio:12.3f}")

    # Per-step displacement distribution
    eqm_all = np.array(eqm_all_disps)
    diff_all = np.array(diff_all_disps)
    print(f"\nPer-step displacement distribution (all steps pooled):")
    print(f"  EqM:     mean={eqm_all.mean():.4f}  std={eqm_all.std():.4f}  "
          f"p50={np.percentile(eqm_all, 50):.4f}  p90={np.percentile(eqm_all, 90):.4f}  "
          f"p99={np.percentile(eqm_all, 99):.4f}  max={eqm_all.max():.4f}")
    print(f"  Diffuser: mean={diff_all.mean():.4f}  std={diff_all.std():.4f}  "
          f"p50={np.percentile(diff_all, 50):.4f}  p90={np.percentile(diff_all, 90):.4f}  "
          f"p99={np.percentile(diff_all, 99):.4f}  max={diff_all.max():.4f}")

    # Conclusion
    print(f"\n{'='*85}")
    eqm_cv_mean = np.mean([s["disp_cv"] for s in eqm_stats])
    diff_cv_mean = np.mean([s["disp_cv"] for s in diff_stats])
    eqm_max_mean = np.mean([s["max_disp"] for s in eqm_stats])
    diff_max_mean = np.mean([s["max_disp"] for s in diff_stats])

    if eqm_cv_mean > diff_cv_mean * 1.5 or eqm_max_mean > diff_max_mean * 2.0:
        print("CONCLUSION: EqM trajectories are numerically LESS smooth than Diffuser.")
        print(f"  Displacement CV ratio: {eqm_cv_mean/diff_cv_mean:.2f}x")
        print(f"  Max displacement ratio: {eqm_max_mean/diff_max_mean:.2f}x")
        print("  → The visual discontinuity is a REAL model property, not a plotting artifact.")
    elif diff_cv_mean > eqm_cv_mean * 1.5:
        print("CONCLUSION: Diffuser trajectories are numerically LESS smooth than EqM.")
        print("  → The visual discontinuity may be a plotting artifact.")
    else:
        print("CONCLUSION: Both models have similar smoothness characteristics.")
        print(f"  Displacement CV ratio: {eqm_cv_mean/diff_cv_mean:.2f}x")
        print(f"  Max displacement ratio: {eqm_max_mean/diff_max_mean:.2f}x")
        print("  → The visual difference may be partly a plotting artifact or a subtle effect.")


if __name__ == "__main__":
    main()
