#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DIFFUSER_PROBE = ROOT / "scripts" / "synthetic_maze2d_diffuser_probe.py"
SAC_PROBE = ROOT / "scripts" / "synthetic_maze2d_sac_her_probe.py"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cfg(mod, cfg_path: Path):
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = mod.Config()
    for k, v in raw.items():
        if v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _pairwise_mean_l2(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    n = int(arr.shape[0])
    if n < 2:
        return float("nan")
    diff = arr[:, None, :] - arr[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    tri = np.triu_indices(n, 1)
    vals = d[tri]
    return float(np.mean(vals)) if vals.size else float("nan")


def _mean_std_dim(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.std(arr, axis=0)))


def _build_queries(dprobe, *, observations: np.ndarray, num_queries: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    bank_size = max(256, int(num_queries))
    bank = dprobe.build_diverse_query_bank(
        points_xy=np.asarray(observations, dtype=np.float32)[:, :2],
        bank_size=int(bank_size),
        n_angle_bins=16,
        min_pair_distance=1.0,
        seed=int(seed) + 991,
    )
    return dprobe.select_query_pairs(query_bank=bank, num_queries=int(num_queries), seed=int(seed) + 7919)


def _build_diffuser(
    dprobe,
    *,
    run_dir: Path,
    ckpt_path: Path,
    device: torch.device,
):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    cfg = _load_cfg(dprobe, cfg_path)

    replay_import = str(getattr(cfg, "replay_import_path", "") or getattr(cfg, "replay_load_npz", "")).strip()
    if replay_import:
        raw_dataset, _action_low, _action_high, _stats, _meta = dprobe.load_replay_artifact(Path(replay_import))
    else:
        raw_dataset, _action_low, _action_high, _stats = dprobe.collect_random_dataset(
            env_name=cfg.env,
            n_episodes=int(cfg.n_episodes),
            episode_len=int(cfg.episode_len),
            action_scale=float(cfg.action_scale),
            seed=int(cfg.seed),
            corridor_aware_data=bool(cfg.corridor_aware_data),
            corridor_max_resamples=int(cfg.corridor_max_resamples),
        )

    dataset, _train_loader, _val_loader, _train_idx, _val_idx = dprobe.build_goal_dataset_splits(
        raw_dataset=raw_dataset,
        cfg=cfg,
        split_seed=int(cfg.seed) + 1337,
        device=device,
    )

    dim_mults = dprobe.parse_dim_mults(cfg.model_dim_mults)
    model = dprobe.TemporalUnet(
        horizon=int(cfg.horizon),
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=dataset.observation_dim,
        dim=int(cfg.model_dim),
        dim_mults=dim_mults,
    ).to(device)
    diffusion = dprobe.GaussianDiffusion(
        model=model,
        horizon=int(cfg.horizon),
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=int(cfg.n_diffusion_steps),
        clip_denoised=bool(cfg.clip_denoised),
        predict_epsilon=bool(cfg.predict_epsilon),
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "ema" in ckpt:
        sampler = copy.deepcopy(diffusion).to(device)
        sampler.load_state_dict(ckpt["ema"])
    elif "model" in ckpt:
        sampler = copy.deepcopy(diffusion).to(device)
        sampler.load_state_dict(ckpt["model"])
    else:
        raise SystemExit(f"Checkpoint {ckpt_path} missing 'ema'/'model' keys")
    sampler.eval()

    return cfg, raw_dataset, dataset, sampler


def _build_sac(
    sprobe,
    dprobe,
    *,
    run_dir: Path,
    ckpt_path: Path,
    device: torch.device,
):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    cfg = _load_cfg(sprobe, cfg_path)

    import gym

    tmp_env = gym.make(cfg.env)
    action_low = np.asarray(tmp_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(tmp_env.action_space.high, dtype=np.float32)
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    act_dim = int(np.prod(tmp_env.action_space.shape))
    tmp_env.close()

    hidden_dims = tuple(int(x) for x in str(cfg.sac_hidden_dims).split(",") if x.strip())
    target_entropy = float(cfg.sac_target_entropy)
    if not np.isfinite(target_entropy):
        target_entropy = -float(act_dim)

    agent = sprobe.SACAgent(
        obs_dim=obs_dim,
        goal_dim=2,
        action_dim=act_dim,
        hidden_dims=hidden_dims,
        log_std_min=float(cfg.sac_log_std_min),
        log_std_max=float(cfg.sac_log_std_max),
        action_low=action_low,
        action_high=action_high,
        lr=float(cfg.learning_rate),
        gamma=float(cfg.gamma),
        tau=float(cfg.tau),
        policy_update_every=int(cfg.policy_update_every),
        target_update_every=int(cfg.target_update_every),
        auto_alpha=bool(cfg.sac_auto_alpha),
        init_alpha=float(cfg.sac_alpha),
        target_entropy=float(target_entropy),
        device=device,
        grad_clip=float(cfg.grad_clip),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    if "actor" not in ckpt:
        raise SystemExit(f"Checkpoint {ckpt_path} missing 'actor' key")
    agent.actor.load_state_dict(ckpt["actor"])
    if "q1" in ckpt:
        agent.q1.load_state_dict(ckpt["q1"])
    if "q2" in ckpt:
        agent.q2.load_state_dict(ckpt["q2"])
    if "q1_targ" in ckpt:
        agent.q1_targ.load_state_dict(ckpt["q1_targ"])
    if "q2_targ" in ckpt:
        agent.q2_targ.load_state_dict(ckpt["q2_targ"])
    if bool(agent.auto_alpha) and ("log_alpha" in ckpt) and (agent.log_alpha is not None):
        with torch.no_grad():
            agent.log_alpha.copy_(torch.tensor(float(ckpt["log_alpha"]), device=device))

    agent.actor.eval()

    # Reuse diffuser helper for replay import (for fair start-goal query support).
    replay_import = str(getattr(cfg, "replay_import_path", "")).strip()
    if replay_import:
        sac_raw_dataset, _a_low, _a_high, _stats, _meta = dprobe.load_replay_artifact(Path(replay_import))
    else:
        sac_raw_dataset, _a_low, _a_high, _stats = dprobe.collect_random_dataset(
            env_name=cfg.env,
            n_episodes=int(cfg.n_episodes),
            episode_len=int(cfg.episode_len),
            action_scale=float(cfg.action_scale),
            seed=int(cfg.seed),
            corridor_aware_data=bool(cfg.corridor_aware_data),
            corridor_max_resamples=int(cfg.corridor_max_resamples),
        )

    return cfg, sac_raw_dataset, agent


def _rollout_diffuser_stochastic(
    dprobe,
    *,
    model,
    dataset,
    env,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    planning_horizon: int,
    rollout_horizon: int,
    replan_every_n_steps: int,
    device: torch.device,
    wall_aware_planning: bool,
    wall_aware_plan_samples: int,
    action_scale_mult: float = 1.0,
    action_ema_beta: float = 0.0,
    adaptive_replan: bool = False,
    adaptive_replan_min: int = 4,
    adaptive_replan_progress_eps: float = 0.01,
    plan_samples: int = 1,
    plan_score_mode: str = "none",
    plan_score_prefix_len: int = -1,
    goal_thresholds: Tuple[float, ...] = (0.1, 0.2),
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    obs = dprobe.reset_rollout_start(env, start_xy=np.asarray(start_xy, dtype=np.float32))
    traj_xy: List[np.ndarray] = [np.asarray(obs[:2], dtype=np.float32)]
    min_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
    final_goal_dist = min_goal_dist
    goal_xy_arr = np.asarray(goal_xy, dtype=np.float32)

    act_low = np.asarray(env.action_space.low, dtype=np.float32)
    act_high = np.asarray(env.action_space.high, dtype=np.float32)
    stride = max(1, int(replan_every_n_steps))

    planned_actions = np.zeros((0, dataset.action_dim), dtype=np.float32)
    plan_offset = 0
    _prev_exec_action = np.zeros(dataset.action_dim, dtype=np.float32)
    _steps_since_replan = 0
    _prev_dist = float(np.linalg.norm(obs[:2] - goal_xy_arr))
    _adaptive_stall_steps = 0

    # Metric accumulators
    _raw_norms: List[float] = []
    _exec_norms: List[float] = []
    _clip_flags: List[int] = []
    _first_hit: Dict[float, Any] = {thr: None for thr in goal_thresholds}

    for t in range(int(rollout_horizon)):
        # Adaptive replan check
        if adaptive_replan and t > 0:
            _progress = _prev_dist - float(np.linalg.norm(obs[:2] - goal_xy_arr))
            if _progress < float(adaptive_replan_progress_eps):
                _adaptive_stall_steps += 1
            else:
                _adaptive_stall_steps = 0
        _adaptive_trigger = (
            adaptive_replan
            and _adaptive_stall_steps >= 1
            and _steps_since_replan >= int(adaptive_replan_min)
        )
        _prev_dist = float(np.linalg.norm(obs[:2] - goal_xy_arr))

        if (t % stride == 0) or (plan_offset >= len(planned_actions)) or _adaptive_trigger:
            _plan_obs, best_actions, _hits = dprobe.sample_best_plan_from_obs(
                model=model,
                dataset=dataset,
                start_obs=obs,
                goal_xy=goal_xy_arr,
                horizon=int(planning_horizon),
                device=device,
                maze_arr=None,
                wall_aware_planning=bool(wall_aware_planning),
                wall_aware_plan_samples=int(wall_aware_plan_samples),
                plan_samples=int(plan_samples),
                plan_score_mode=str(plan_score_mode),
                plan_score_prefix_len=int(plan_score_prefix_len),
                replan_every=stride,
            )
            planned_actions = np.asarray(best_actions, dtype=np.float32)
            plan_offset = 0
            _steps_since_replan = 0
            if _adaptive_trigger:
                _adaptive_stall_steps = 0

        _steps_since_replan += 1
        action_raw = (
            planned_actions[plan_offset] if plan_offset < len(planned_actions)
            else np.zeros(dataset.action_dim, dtype=np.float32)
        ).astype(np.float32)
        plan_offset += 1

        # Action scaling + EMA transform (RANK 2)
        act_scaled = np.clip(float(action_scale_mult) * action_raw, act_low, act_high)
        if t == 0 or float(action_ema_beta) == 0.0:
            action = act_scaled.astype(np.float32)
        else:
            action = np.clip(
                (1.0 - float(action_ema_beta)) * act_scaled + float(action_ema_beta) * _prev_exec_action,
                act_low, act_high,
            ).astype(np.float32)
        _prev_exec_action = action.copy()

        # Track raw norms and clip flag
        _raw_norms.append(float(np.linalg.norm(action_raw)))
        _exec_norms.append(float(np.linalg.norm(action)))
        scaled_pre_clip = float(action_scale_mult) * action_raw
        clipped = int(np.any(scaled_pre_clip < act_low) or np.any(scaled_pre_clip > act_high))
        _clip_flags.append(clipped)

        obs, _r, _d, _info = dprobe.safe_step(env, action)
        traj_xy.append(np.asarray(obs[:2], dtype=np.float32))
        dist = float(np.linalg.norm(obs[:2] - goal_xy_arr))
        min_goal_dist = min(min_goal_dist, dist)
        final_goal_dist = dist
        for thr in goal_thresholds:
            if _first_hit[thr] is None and dist <= thr:
                _first_hit[thr] = t

    extra: Dict[str, Any] = {
        "mean_action_l2_raw": float(np.mean(_raw_norms)) if _raw_norms else float("nan"),
        "mean_action_l2_exec": float(np.mean(_exec_norms)) if _exec_norms else float("nan"),
        "clip_fraction": float(np.mean(_clip_flags)) if _clip_flags else float("nan"),
    }
    for thr in goal_thresholds:
        key = f"steps_to_threshold_{str(thr).replace('.', 'p')}"
        extra[key] = _first_hit[thr]

    return np.stack(traj_xy, axis=0), float(min_goal_dist), float(final_goal_dist), extra


def _rollout_sac_stochastic(
    dprobe,
    *,
    agent,
    env,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rollout_horizon: int,
    decision_every_n_steps: int,
) -> Tuple[np.ndarray, float, float]:
    obs = dprobe.reset_rollout_start(env, start_xy=np.asarray(start_xy, dtype=np.float32))
    traj_xy: List[np.ndarray] = [np.asarray(obs[:2], dtype=np.float32)]
    min_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
    final_goal_dist = min_goal_dist

    act_low = np.asarray(env.action_space.low, dtype=np.float32)
    act_high = np.asarray(env.action_space.high, dtype=np.float32)
    stride = max(1, int(decision_every_n_steps))

    cached_action = None
    for t in range(int(rollout_horizon)):
        if cached_action is None or (t % stride == 0):
            cached_action = agent.act(obs, np.asarray(goal_xy, dtype=np.float32), deterministic=False)
        action = np.clip(np.asarray(cached_action, dtype=np.float32), act_low, act_high)
        obs, _r, _d, _info = dprobe.safe_step(env, action)
        traj_xy.append(np.asarray(obs[:2], dtype=np.float32))
        dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = min(min_goal_dist, dist)
        final_goal_dist = dist

    return np.stack(traj_xy, axis=0), float(min_goal_dist), float(final_goal_dist)


def _plot_query_rollout_compare(
    dprobe,
    *,
    out_path: Path,
    query_id: int,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    diffuser_trajs: Sequence[np.ndarray],
    sac_trajs: Sequence[np.ndarray],
    maze_arr: np.ndarray | None,
    diffuser_success_rate: float,
    sac_success_rate: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), squeeze=False)
    for ax, title, trajs, color, succ in [
        (axes[0, 0], "Diffuser rollouts", diffuser_trajs, "#1f77b4", diffuser_success_rate),
        (axes[0, 1], "SAC rollouts", sac_trajs, "#d62728", sac_success_rate),
    ]:
        dprobe.draw_maze_geometry(ax, maze_arr=maze_arr)
        for xy in trajs:
            arr = np.asarray(xy, dtype=np.float32)
            ax.plot(arr[:, 0], arr[:, 1], color=color, alpha=0.35, linewidth=1.7)
        line = np.stack([np.asarray(start_xy, dtype=np.float32), np.asarray(goal_xy, dtype=np.float32)], axis=0)
        ax.plot(line[:, 0], line[:, 1], "--", color="black", linewidth=1.2, alpha=0.7)
        ax.scatter([start_xy[0]], [start_xy[1]], c="black", s=40, marker="o", label="start")
        ax.scatter([goal_xy[0]], [goal_xy[1]], c="#2ca02c", s=85, marker="*", label="goal")
        ax.set_title(f"{title} | success={succ:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Query {query_id}: same start-goal, stochastic rollout samples", fontsize=12)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles, labels, loc="best", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_query_overlay_grid(
    dprobe,
    *,
    out_path: Path,
    records: Sequence[Dict[str, Any]],
    maze_arr: np.ndarray | None,
    max_queries: int,
) -> None:
    if not records:
        return
    records = list(records)[: max(1, int(max_queries))]
    n = len(records)
    cols = min(3, n)
    rows = int(math.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.6 * rows), squeeze=False)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx >= n:
            ax.axis("off")
            continue
        rec = records[idx]
        dprobe.draw_maze_geometry(ax, maze_arr=maze_arr)
        for xy in rec["diffuser_trajs"]:
            arr = np.asarray(xy, dtype=np.float32)
            ax.plot(arr[:, 0], arr[:, 1], color="#1f77b4", alpha=0.28, linewidth=1.6)
        for xy in rec["sac_trajs"]:
            arr = np.asarray(xy, dtype=np.float32)
            ax.plot(arr[:, 0], arr[:, 1], color="#d62728", alpha=0.28, linewidth=1.6)
        start_xy = np.asarray(rec["start_xy"], dtype=np.float32)
        goal_xy = np.asarray(rec["goal_xy"], dtype=np.float32)
        line = np.stack([start_xy, goal_xy], axis=0)
        ax.plot(line[:, 0], line[:, 1], "--", color="black", linewidth=1.0, alpha=0.6)
        ax.scatter([start_xy[0]], [start_xy[1]], c="black", s=30, marker="o")
        ax.scatter([goal_xy[0]], [goal_xy[1]], c="#2ca02c", s=75, marker="*")
        ax.set_title(
            f"Q{rec['query_id']} | succ D={rec['diffuser_success_rate']:.2f} S={rec['sac_success_rate']:.2f}",
            fontsize=10,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=2, label="Diffuser"),
        plt.Line2D([0], [0], color="#d62728", lw=2, label="SAC"),
        plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="start-goal line"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Trajectory overlays for identical start-goal queries", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare collector stochasticity: diffuser denoising vs SAC policy sampling.")
    ap.add_argument("--diffuser-run-dir", type=Path, required=True)
    ap.add_argument("--sac-run-dir", type=Path, default=None, help="Optional: SAC run dir for comparison (omit for Diffuser-only mode).")
    ap.add_argument("--diffuser-checkpoint", type=Path, default=None)
    ap.add_argument("--sac-checkpoint", type=Path, default=None)
    ap.add_argument("--num-queries", type=int, default=12)
    ap.add_argument("--samples-per-query", type=int, default=50, help="Action-sample count per start-goal for stochasticity stats.")
    ap.add_argument("--rollouts-per-query", type=int, default=16, help="Repeated env rollouts per start-goal for endpoint diversity stats.")
    ap.add_argument("--planning-horizon", type=int, default=None)
    ap.add_argument("--rollout-horizon", type=int, default=256)
    ap.add_argument("--diffuser-replan-every", type=int, default=16)
    ap.add_argument("--sac-decision-every", type=int, default=16)
    ap.add_argument("--goal-success-threshold", type=float, default=0.2)
    ap.add_argument("--save-trajectory-plots", action="store_true", help="Save per-query rollout trajectory comparison plots.")
    ap.add_argument("--plot-rollouts-per-query", type=int, default=8, help="How many stochastic rollouts per method to draw for each query.")
    ap.add_argument("--plot-max-queries", type=int, default=12, help="Max number of queries included in the summary grid plot.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=None)
    # --- Execution-time action transform (RANK 2) ---
    ap.add_argument("--diffuser-action-scale-mult", type=float, default=1.0,
                    help="Multiply Diffuser actions by this scalar before clipping (1.0 = off).")
    ap.add_argument("--diffuser-action-ema-beta", type=float, default=0.0,
                    help="EMA smoothing on Diffuser executed actions (0.0 = off).")
    # --- Adaptive replanning (RANK 4) ---
    ap.add_argument("--adaptive-replan", action="store_true", default=False,
                    help="Enable adaptive replanning when progress stalls.")
    ap.add_argument("--adaptive-replan-min", type=int, default=4,
                    help="Minimum steps between adaptive replans.")
    ap.add_argument("--adaptive-replan-progress-eps", type=float, default=0.01,
                    help="Replan early if per-step distance reduction < this value.")
    # --- Prefix-progress plan scoring (RANK 1) ---
    ap.add_argument("--plan-samples", type=int, default=1,
                    help="Number of imagined plans per replan to score and select from.")
    ap.add_argument("--plan-score-mode", type=str, default="none",
                    choices=["none", "min_dist_prefix", "dist_at_L"],
                    help="Prefix scoring mode for best-of-K plan selection.")
    ap.add_argument("--plan-score-prefix-len", type=int, default=-1,
                    help="Prefix length for scoring (-1 = replan_every steps).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise SystemExit("--num-queries must be > 0")
    if args.samples_per_query <= 1:
        raise SystemExit("--samples-per-query must be > 1")
    if args.rollouts_per_query <= 1:
        raise SystemExit("--rollouts-per-query must be > 1")
    if args.plot_rollouts_per_query <= 0:
        raise SystemExit("--plot-rollouts-per-query must be > 0")

    mujoco_bin = "/root/.mujoco/mujoco210/bin"
    mujoco_compat_dir = Path("/tmp/mujoco_compat")
    mujoco_compat_dir.mkdir(parents=True, exist_ok=True)
    glew_src = Path("/usr/lib/x86_64-linux-gnu/libGLEW.so")
    glew_compat = mujoco_compat_dir / "libglewosmesa.so"
    if glew_src.exists() and not glew_compat.exists():
        try:
            glew_compat.symlink_to(glew_src)
        except FileExistsError:
            pass

    ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [p for p in ld_lib_path.split(":") if p]
    if str(mujoco_compat_dir) not in ld_parts:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{mujoco_compat_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            if os.environ.get("LD_LIBRARY_PATH")
            else str(mujoco_compat_dir)
        )
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [p for p in ld_lib_path.split(":") if p]
    if mujoco_bin not in ld_parts:
        os.environ["LD_LIBRARY_PATH"] = f"{ld_lib_path}:{mujoco_bin}" if ld_lib_path else mujoco_bin
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    sys.path.insert(0, str(ROOT / "third_party" / "diffuser-maze2d"))
    sys.path.insert(0, str(ROOT / "scripts"))

    # d4rl catches ImportError for optional env packs, but Gym may raise
    # DependencyNotInstalled (not a subclass of ImportError). Re-map so optional
    # imports are treated consistently.
    import gym

    gym.error.DependencyNotInstalled = ImportError

    dprobe = _load_module(DIFFUSER_PROBE, "dprobe_stoch")
    sprobe = _load_module(SAC_PROBE, "sprobe_stoch")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    diffuser_run = args.diffuser_run_dir.resolve()
    diffuser_ckpt = args.diffuser_checkpoint.resolve() if args.diffuser_checkpoint else (diffuser_run / "checkpoint_last.pt")

    dcfg, d_raw, d_dataset, d_model = _build_diffuser(
        dprobe,
        run_dir=diffuser_run,
        ckpt_path=diffuser_ckpt,
        device=device,
    )

    _sac_enabled = args.sac_run_dir is not None
    if _sac_enabled:
        sac_run = args.sac_run_dir.resolve()
        sac_ckpt = args.sac_checkpoint.resolve() if args.sac_checkpoint else (sac_run / "checkpoint_last.pt")
        scfg, s_raw, s_agent = _build_sac(
            sprobe,
            dprobe,
            run_dir=sac_run,
            ckpt_path=sac_ckpt,
            device=device,
        )
    else:
        scfg = dcfg
        s_agent = None

    # Prefer query source from diffuser replay for consistency with diffuser collector settings.
    query_pairs = _build_queries(
        dprobe,
        observations=d_raw["observations"],
        num_queries=int(args.num_queries),
        seed=int(args.seed),
    )

    import gym

    d_env = gym.make(str(dcfg.env))
    s_env = gym.make(str(scfg.env)) if _sac_enabled else None
    maze_arr = dprobe.load_maze_arr_from_env(str(dcfg.env))

    planning_horizon = int(args.planning_horizon) if args.planning_horizon is not None else int(dcfg.horizon)
    out_dir = args.out_dir.resolve() if args.out_dir else (ROOT / "runs" / "analysis" / "collector_stochasticity" / f"collector_stochasticity_{_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "trajectory_plots"
    if args.save_trajectory_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    plot_records: List[Dict[str, Any]] = []

    for qid, (start_xy, goal_xy) in enumerate(query_pairs):
        # Diffuser action stochasticity from denoising samples.
        d_obs_samples, d_act_samples = dprobe.sample_imagined_trajectory(
            model=d_model,
            dataset=d_dataset,
            start_xy=start_xy,
            goal_xy=goal_xy,
            horizon=int(planning_horizon),
            device=device,
            n_samples=int(args.samples_per_query),
        )
        d_a0 = np.asarray(d_act_samples[:, 0, :], dtype=np.float32)

        # SAC action stochasticity from policy sampling at same start observation.
        if _sac_enabled:
            s_obs0 = dprobe.reset_rollout_start(s_env, start_xy=start_xy)
            s_a0 = np.stack(
                [
                    s_agent.act(np.asarray(s_obs0, dtype=np.float32), np.asarray(goal_xy, dtype=np.float32), deterministic=False)
                    for _ in range(int(args.samples_per_query))
                ],
                axis=0,
            ).astype(np.float32)
        else:
            s_a0 = np.zeros((int(args.samples_per_query), 2), dtype=np.float32)

        # Rollout diversity (stochastic collector behavior).
        d_end_xy: List[np.ndarray] = []
        d_min_d: List[float] = []
        s_end_xy: List[np.ndarray] = []
        s_min_d: List[float] = []
        d_plot_trajs: List[np.ndarray] = []
        s_plot_trajs: List[np.ndarray] = []
        n_plot_rollouts = min(int(args.plot_rollouts_per_query), int(args.rollouts_per_query))
        d_mean_l2_raw_list: List[float] = []
        d_clip_frac_list: List[float] = []
        d_steps_0p1_list: List[Any] = []
        d_steps_0p2_list: List[Any] = []
        for _ in range(int(args.rollouts_per_query)):
            d_traj, d_min, _d_fin, d_extra = _rollout_diffuser_stochastic(
                dprobe,
                model=d_model,
                dataset=d_dataset,
                env=d_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                planning_horizon=int(planning_horizon),
                rollout_horizon=int(args.rollout_horizon),
                replan_every_n_steps=int(args.diffuser_replan_every),
                device=device,
                wall_aware_planning=bool(getattr(dcfg, "wall_aware_planning", False)),
                wall_aware_plan_samples=int(getattr(dcfg, "wall_aware_plan_samples", 1)),
                action_scale_mult=float(args.diffuser_action_scale_mult),
                action_ema_beta=float(args.diffuser_action_ema_beta),
                adaptive_replan=bool(args.adaptive_replan),
                adaptive_replan_min=int(args.adaptive_replan_min),
                adaptive_replan_progress_eps=float(args.adaptive_replan_progress_eps),
                plan_samples=int(args.plan_samples),
                plan_score_mode=str(args.plan_score_mode),
                plan_score_prefix_len=int(args.plan_score_prefix_len),
            )
            d_mean_l2_raw_list.append(d_extra.get("mean_action_l2_raw", float("nan")))
            d_clip_frac_list.append(d_extra.get("clip_fraction", float("nan")))
            d_steps_0p1_list.append(d_extra.get("steps_to_threshold_0p1"))
            d_steps_0p2_list.append(d_extra.get("steps_to_threshold_0p2"))
            d_end_xy.append(np.asarray(d_traj[-1], dtype=np.float32))
            d_min_d.append(float(d_min))
            if args.save_trajectory_plots and len(d_plot_trajs) < n_plot_rollouts:
                d_plot_trajs.append(np.asarray(d_traj, dtype=np.float32))

            if _sac_enabled:
                s_traj, s_min, _s_fin = _rollout_sac_stochastic(
                    dprobe,
                    agent=s_agent,
                    env=s_env,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    rollout_horizon=int(args.rollout_horizon),
                    decision_every_n_steps=int(args.sac_decision_every),
                )
                s_end_xy.append(np.asarray(s_traj[-1], dtype=np.float32))
                s_min_d.append(float(s_min))
                if args.save_trajectory_plots and len(s_plot_trajs) < n_plot_rollouts:
                    s_plot_trajs.append(np.asarray(s_traj, dtype=np.float32))
            else:
                # SAC not available; fill with NaN sentinel
                s_end_xy.append(np.full(2, float("nan"), dtype=np.float32))
                s_min_d.append(float("nan"))

        d_end_xy_np = np.asarray(d_end_xy, dtype=np.float32)
        s_end_xy_np = np.asarray(s_end_xy, dtype=np.float32)

        d_succ = float(np.mean(np.asarray(d_min_d) <= float(args.goal_success_threshold)))
        s_succ = float(np.mean(np.asarray(s_min_d) <= float(args.goal_success_threshold)))

        # Aggregate new Diffuser-only metrics over rollouts
        d_mean_l2_raw = float(np.nanmean(d_mean_l2_raw_list)) if d_mean_l2_raw_list else float("nan")
        d_clip_frac = float(np.nanmean(d_clip_frac_list)) if d_clip_frac_list else float("nan")
        _valid_0p1 = [x for x in d_steps_0p1_list if x is not None]
        _valid_0p2 = [x for x in d_steps_0p2_list if x is not None]
        d_hit_rate_0p1 = float(len(_valid_0p1)) / len(d_steps_0p1_list) if d_steps_0p1_list else float("nan")
        d_hit_rate_0p2 = float(len(_valid_0p2)) / len(d_steps_0p2_list) if d_steps_0p2_list else float("nan")
        d_steps_mean_0p1 = float(np.mean(_valid_0p1)) if _valid_0p1 else float("nan")
        d_steps_mean_0p2 = float(np.mean(_valid_0p2)) if _valid_0p2 else float("nan")

        rows.append(
            {
                "query_id": int(qid),
                "start_x": float(start_xy[0]),
                "start_y": float(start_xy[1]),
                "goal_x": float(goal_xy[0]),
                "goal_y": float(goal_xy[1]),
                "diffuser_action_std_mean": _mean_std_dim(d_a0),
                "diffuser_action_pairwise_l2_mean": _pairwise_mean_l2(d_a0),
                "sac_action_std_mean": _mean_std_dim(s_a0),
                "sac_action_pairwise_l2_mean": _pairwise_mean_l2(s_a0),
                "diffuser_rollout_endpoint_std_mean": _mean_std_dim(d_end_xy_np),
                "diffuser_rollout_endpoint_pairwise_l2_mean": _pairwise_mean_l2(d_end_xy_np),
                "sac_rollout_endpoint_std_mean": _mean_std_dim(s_end_xy_np),
                "sac_rollout_endpoint_pairwise_l2_mean": _pairwise_mean_l2(s_end_xy_np),
                "diffuser_rollout_success_rate": d_succ,
                "sac_rollout_success_rate": s_succ,
                "diffuser_rollout_min_goal_dist_mean": float(np.mean(d_min_d)),
                "sac_rollout_min_goal_dist_mean": float(np.mean(s_min_d)),
                # New Diffuser-only metrics (RANK 2 / F)
                "diffuser_mean_action_l2_raw": d_mean_l2_raw,
                "diffuser_clip_fraction": d_clip_frac,
                "diffuser_hit_rate_0p1": d_hit_rate_0p1,
                "diffuser_hit_rate_0p2": d_hit_rate_0p2,
                "diffuser_steps_to_0p1_mean": d_steps_mean_0p1,
                "diffuser_steps_to_0p2_mean": d_steps_mean_0p2,
            }
        )
        if args.save_trajectory_plots and d_plot_trajs and s_plot_trajs:
            q_plot = plots_dir / f"query_{int(qid):02d}_sac_vs_diffuser.png"
            _plot_query_rollout_compare(
                dprobe,
                out_path=q_plot,
                query_id=int(qid),
                start_xy=np.asarray(start_xy, dtype=np.float32),
                goal_xy=np.asarray(goal_xy, dtype=np.float32),
                diffuser_trajs=d_plot_trajs,
                sac_trajs=s_plot_trajs,
                maze_arr=maze_arr,
                diffuser_success_rate=d_succ,
                sac_success_rate=s_succ,
            )
            npz_path = plots_dir / f"query_{int(qid):02d}_trajectories.npz"
            np.savez_compressed(
                npz_path,
                query_id=int(qid),
                start_xy=np.asarray(start_xy, dtype=np.float32),
                goal_xy=np.asarray(goal_xy, dtype=np.float32),
                diffuser_trajs=np.stack(d_plot_trajs, axis=0),
                sac_trajs=np.stack(s_plot_trajs, axis=0),
            )
            plot_records.append(
                {
                    "query_id": int(qid),
                    "start_xy": np.asarray(start_xy, dtype=np.float32),
                    "goal_xy": np.asarray(goal_xy, dtype=np.float32),
                    "diffuser_success_rate": d_succ,
                    "sac_success_rate": s_succ,
                    "diffuser_trajs": d_plot_trajs,
                    "sac_trajs": s_plot_trajs,
                }
            )

    d_env.close()
    if s_env is not None:
        s_env.close()

    df = pd.DataFrame(rows)
    csv_path = out_dir / "collector_stochasticity.csv"
    df.to_csv(csv_path, index=False)

    def _col_mean(col: str) -> float:
        return float(df[col].mean()) if len(df) and col in df.columns else float("nan")

    agg = {
        "queries": int(len(df)),
        "diffuser_action_std_mean": _col_mean("diffuser_action_std_mean"),
        "sac_action_std_mean": _col_mean("sac_action_std_mean"),
        "diffuser_action_pairwise_l2_mean": _col_mean("diffuser_action_pairwise_l2_mean"),
        "sac_action_pairwise_l2_mean": _col_mean("sac_action_pairwise_l2_mean"),
        "diffuser_rollout_endpoint_pairwise_l2_mean": _col_mean("diffuser_rollout_endpoint_pairwise_l2_mean"),
        "sac_rollout_endpoint_pairwise_l2_mean": _col_mean("sac_rollout_endpoint_pairwise_l2_mean"),
        "diffuser_rollout_success_rate_mean": _col_mean("diffuser_rollout_success_rate"),
        "sac_rollout_success_rate_mean": _col_mean("sac_rollout_success_rate"),
        "diffuser_rollout_min_goal_dist_mean": _col_mean("diffuser_rollout_min_goal_dist_mean"),
        # New metrics (RANK 2 / F)
        "diffuser_mean_action_l2_raw": _col_mean("diffuser_mean_action_l2_raw"),
        "diffuser_clip_fraction": _col_mean("diffuser_clip_fraction"),
        "diffuser_hit_rate_0p1": _col_mean("diffuser_hit_rate_0p1"),
        "diffuser_hit_rate_0p2": _col_mean("diffuser_hit_rate_0p2"),
        "diffuser_steps_to_0p1_mean": _col_mean("diffuser_steps_to_0p1_mean"),
        "diffuser_steps_to_0p2_mean": _col_mean("diffuser_steps_to_0p2_mean"),
        # Config echoed for traceability
        "action_scale_mult": float(args.diffuser_action_scale_mult),
        "action_ema_beta": float(args.diffuser_action_ema_beta),
        "adaptive_replan": bool(args.adaptive_replan),
        "plan_samples": int(args.plan_samples),
        "plan_score_mode": str(args.plan_score_mode),
    }

    summary_path = out_dir / "collector_stochasticity_summary.json"
    summary_path.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Collector Stochasticity Report",
        "",
        f"Created: {datetime.now().isoformat()}",
        f"Diffuser run: {diffuser_run}",
        f"SAC run: {sac_run}",
        f"Queries: {agg['queries']}",
        f"Samples/query (action): {int(args.samples_per_query)}",
        f"Rollouts/query: {int(args.rollouts_per_query)}",
        "",
        "## Aggregate",
        "",
        "| metric | diffuser | sac |",
        "|---|---:|---:|",
        f"| action_std_mean | {agg['diffuser_action_std_mean']:.6f} | {agg['sac_action_std_mean']:.6f} |",
        f"| action_pairwise_l2_mean | {agg['diffuser_action_pairwise_l2_mean']:.6f} | {agg['sac_action_pairwise_l2_mean']:.6f} |",
        f"| rollout_endpoint_pairwise_l2_mean | {agg['diffuser_rollout_endpoint_pairwise_l2_mean']:.6f} | {agg['sac_rollout_endpoint_pairwise_l2_mean']:.6f} |",
        f"| rollout_success_rate_mean | {agg['diffuser_rollout_success_rate_mean']:.6f} | {agg['sac_rollout_success_rate_mean']:.6f} |",
        "",
        "## Outputs",
        "",
        f"- `{csv_path}`",
        f"- `{summary_path}`",
    ]
    if args.save_trajectory_plots and plot_records:
        grid_path = plots_dir / "trajectory_overlay_grid.png"
        _plot_query_overlay_grid(
            dprobe,
            out_path=grid_path,
            records=plot_records,
            maze_arr=maze_arr,
            max_queries=int(args.plot_max_queries),
        )
        md_lines.extend(
            [
                f"- `{plots_dir}`",
                f"- `{grid_path}`",
            ]
        )
    md_path = out_dir / "collector_stochasticity_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] csv={csv_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] report={md_path}")


if __name__ == "__main__":
    main()
