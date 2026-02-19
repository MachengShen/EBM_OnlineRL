#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ebm_online_rl.envs import PointMass2D
from ebm_online_rl.online import GaussianDiffusion1D, TemporalUNet1D, plan_action


MAZE2D_REFERENCE = {
    "base_dim": 32,
    "dim_mults": (1, 4, 8),
    "kernel_size": 5,
    "predict_epsilon": False,
    "horizon_default": 256,
    "horizon_umaze": 128,
    "n_diffusion_steps_default": 256,
    "n_diffusion_steps_umaze": 64,
}


@dataclass
class TrajectoryStats:
    states: np.ndarray
    actions: np.ndarray
    state_action_pairs: np.ndarray
    final_distances: np.ndarray
    min_distances: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight diagnostics for PointMass2D diffusion planner.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint step_*.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_episodes", type=int, default=80)
    parser.add_argument("--episode_len", type=int, default=-1, help="Override episode length; -1 uses checkpoint config.")
    parser.add_argument("--bins", type=int, default=24, help="2D bins for state/action coverage heatmaps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default="")
    return parser.parse_args()


def choose_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but CUDA is unavailable.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return device


def parse_dim_mults(value: str) -> Tuple[int, ...]:
    dim_mults = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not dim_mults:
        raise ValueError("model_dim_mults cannot be empty.")
    return dim_mults


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[GaussianDiffusion1D, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg:
        raise ValueError(f"No config found in checkpoint: {checkpoint_path}")

    env = PointMass2D()
    transition_dim = env.obs_dim + env.act_dim
    dim_mults = parse_dim_mults(str(cfg["model_dim_mults"]))
    denoiser = TemporalUNet1D(
        transition_dim=transition_dim,
        base_dim=int(cfg["model_base_dim"]),
        dim_mults=dim_mults,
    )
    model = GaussianDiffusion1D(
        model=denoiser,
        horizon=int(cfg["horizon"]),
        transition_dim=transition_dim,
        action_dim=env.act_dim,
        n_diffusion_steps=int(cfg["n_diffusion_steps"]),
        predict_epsilon=bool(cfg.get("predict_epsilon", True)),
        clip_denoised=bool(cfg.get("clip_denoised", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def run_rollouts(
    model: GaussianDiffusion1D,
    policy_mode: str,
    env: PointMass2D,
    n_episodes: int,
    rng: np.random.Generator,
    device: torch.device,
) -> TrajectoryStats:
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_sa: List[np.ndarray] = []
    final_dists: List[float] = []
    min_dists: List[float] = []

    for _ in range(n_episodes):
        goal = env.sample_goal(rng)
        obs = env.reset(goal=goal)
        states = [obs.copy()]
        actions = []
        done = False
        min_dist = float(np.linalg.norm(obs - goal))
        final_dist = min_dist

        while not done:
            if policy_mode == "random":
                action = rng.uniform(-env.action_limit, env.action_limit, size=(env.act_dim,)).astype(np.float32)
            elif policy_mode == "planner":
                action = plan_action(
                    model=model,
                    obs=obs,
                    goal=goal,
                    obs_dim=env.obs_dim,
                    act_dim=env.act_dim,
                    action_scale=env.action_limit,
                    device=device,
                    check_conditioning=False,
                )
            else:
                raise ValueError(f"Unknown policy mode: {policy_mode}")

            prev_obs = obs.copy()
            obs, _, done, info = env.step(action)
            states.append(obs.copy())
            actions.append(action.copy())
            all_sa.append(np.concatenate([prev_obs, action], axis=0))

            dist = float(info["dist_to_goal"])
            min_dist = min(min_dist, dist)
            final_dist = dist

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        all_states.append(states_np)
        all_actions.append(actions_np)
        final_dists.append(final_dist)
        min_dists.append(min_dist)

    return TrajectoryStats(
        states=np.concatenate(all_states, axis=0),
        actions=np.concatenate(all_actions, axis=0),
        state_action_pairs=np.asarray(all_sa, dtype=np.float32),
        final_distances=np.asarray(final_dists, dtype=np.float32),
        min_distances=np.asarray(min_dists, dtype=np.float32),
    )


def normalized_entropy(hist: np.ndarray) -> float:
    flat = hist.astype(np.float64).reshape(-1)
    total = flat.sum()
    if total <= 0:
        return 0.0
    probs = flat / total
    nz = probs[probs > 0]
    entropy = -float(np.sum(nz * np.log(nz)))
    max_entropy = math.log(float(len(flat)))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def coverage_metrics(stats: TrajectoryStats, env: PointMass2D, bins: int) -> Dict[str, float]:
    state_hist, _, _ = np.histogram2d(
        stats.states[:, 0],
        stats.states[:, 1],
        bins=bins,
        range=[[-env.state_limit, env.state_limit], [-env.state_limit, env.state_limit]],
    )
    action_hist, _, _ = np.histogram2d(
        stats.actions[:, 0],
        stats.actions[:, 1],
        bins=bins,
        range=[[-env.action_limit, env.action_limit], [-env.action_limit, env.action_limit]],
    )

    sa_bins = max(8, bins // 3)
    sa_hist, _ = np.histogramdd(
        stats.state_action_pairs,
        bins=(sa_bins, sa_bins, sa_bins, sa_bins),
        range=[
            [-env.state_limit, env.state_limit],
            [-env.state_limit, env.state_limit],
            [-env.action_limit, env.action_limit],
            [-env.action_limit, env.action_limit],
        ],
    )

    metrics = {
        "state_coverage_ratio": float(np.count_nonzero(state_hist) / state_hist.size),
        "state_entropy_norm": float(normalized_entropy(state_hist)),
        "action_coverage_ratio": float(np.count_nonzero(action_hist) / action_hist.size),
        "action_entropy_norm": float(normalized_entropy(action_hist)),
        "state_action_coverage_ratio": float(np.count_nonzero(sa_hist) / sa_hist.size),
        "final_dist_mean": float(np.mean(stats.final_distances)),
        "final_dist_std": float(np.std(stats.final_distances)),
        "min_dist_mean": float(np.mean(stats.min_distances)),
        "min_dist_std": float(np.std(stats.min_distances)),
        "success_at_005": float(np.mean(stats.min_distances <= 0.05)),
        "success_at_010": float(np.mean(stats.min_distances <= 0.10)),
        "success_at_020": float(np.mean(stats.min_distances <= 0.20)),
    }
    return metrics


def plot_heatmaps(
    random_stats: TrajectoryStats,
    planner_stats: TrajectoryStats,
    random_metrics: Dict[str, float],
    planner_metrics: Dict[str, float],
    env: PointMass2D,
    bins: int,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    random_state_hist, _, _ = np.histogram2d(
        random_stats.states[:, 0],
        random_stats.states[:, 1],
        bins=bins,
        range=[[-env.state_limit, env.state_limit], [-env.state_limit, env.state_limit]],
    )
    planner_state_hist, _, _ = np.histogram2d(
        planner_stats.states[:, 0],
        planner_stats.states[:, 1],
        bins=bins,
        range=[[-env.state_limit, env.state_limit], [-env.state_limit, env.state_limit]],
    )
    random_action_hist, _, _ = np.histogram2d(
        random_stats.actions[:, 0],
        random_stats.actions[:, 1],
        bins=bins,
        range=[[-env.action_limit, env.action_limit], [-env.action_limit, env.action_limit]],
    )
    planner_action_hist, _, _ = np.histogram2d(
        planner_stats.actions[:, 0],
        planner_stats.actions[:, 1],
        bins=bins,
        range=[[-env.action_limit, env.action_limit], [-env.action_limit, env.action_limit]],
    )

    extent_state = [-env.state_limit, env.state_limit, -env.state_limit, env.state_limit]
    extent_action = [-env.action_limit, env.action_limit, -env.action_limit, env.action_limit]

    im0 = axes[0, 0].imshow(random_state_hist.T, origin="lower", extent=extent_state, aspect="equal")
    axes[0, 0].set_title(
        f"Random State\ncov={random_metrics['state_coverage_ratio']:.3f}, H={random_metrics['state_entropy_norm']:.3f}"
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(planner_state_hist.T, origin="lower", extent=extent_state, aspect="equal")
    axes[0, 1].set_title(
        f"Planner State\ncov={planner_metrics['state_coverage_ratio']:.3f}, H={planner_metrics['state_entropy_norm']:.3f}"
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(random_action_hist.T, origin="lower", extent=extent_action, aspect="equal")
    axes[1, 0].set_title(
        f"Random Action\ncov={random_metrics['action_coverage_ratio']:.3f}, H={random_metrics['action_entropy_norm']:.3f}"
    )
    axes[1, 0].set_xlabel("ax")
    axes[1, 0].set_ylabel("ay")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(planner_action_hist.T, origin="lower", extent=extent_action, aspect="equal")
    axes[1, 1].set_title(
        f"Planner Action\ncov={planner_metrics['action_coverage_ratio']:.3f}, H={planner_metrics['action_entropy_norm']:.3f}"
    )
    axes[1, 1].set_xlabel("ax")
    axes[1, 1].set_ylabel("ay")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def architecture_summary(model: GaussianDiffusion1D, cfg: Dict) -> Dict:
    denoiser = model.model
    kernel_size = int(denoiser.downs[0]["res1"].block1.net[0].kernel_size[0])
    sequence_length = int(model.sequence_length)
    lengths = [sequence_length]
    for block in denoiser.downs:
        if isinstance(block["down"], torch.nn.Identity):
            break
        lengths.append((lengths[-1] + 1) // 2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dim_mults = parse_dim_mults(str(cfg["model_dim_mults"]))

    return {
        "base_dim": int(cfg["model_base_dim"]),
        "dim_mults": dim_mults,
        "kernel_size": kernel_size,
        "horizon": int(cfg["horizon"]),
        "sequence_lengths_downpath": lengths,
        "n_diffusion_steps": int(cfg["n_diffusion_steps"]),
        "predict_epsilon": bool(cfg.get("predict_epsilon", True)),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def make_comparison(current: Dict) -> Dict:
    return {
        "base_dim_matches_maze2d": bool(current["base_dim"] == MAZE2D_REFERENCE["base_dim"]),
        "dim_mults_matches_maze2d": bool(tuple(current["dim_mults"]) == MAZE2D_REFERENCE["dim_mults"]),
        "kernel_size_matches_maze2d": bool(current["kernel_size"] == MAZE2D_REFERENCE["kernel_size"]),
        "predict_epsilon_matches_maze2d": bool(current["predict_epsilon"] == MAZE2D_REFERENCE["predict_epsilon"]),
        "horizon_vs_umaze_ratio": float(current["horizon"] / MAZE2D_REFERENCE["horizon_umaze"]),
        "horizon_vs_default_ratio": float(current["horizon"] / MAZE2D_REFERENCE["horizon_default"]),
        "diffusion_steps_vs_umaze_ratio": float(
            current["n_diffusion_steps"] / MAZE2D_REFERENCE["n_diffusion_steps_umaze"]
        ),
        "diffusion_steps_vs_default_ratio": float(
            current["n_diffusion_steps"] / MAZE2D_REFERENCE["n_diffusion_steps_default"]
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(args.device)
    model, cfg = load_model(checkpoint_path, device=device)

    episode_len = int(cfg.get("episode_len", 50)) if args.episode_len < 0 else int(args.episode_len)
    env = PointMass2D(episode_length=episode_len)
    rng = np.random.default_rng(args.seed)

    random_stats = run_rollouts(
        model=model,
        policy_mode="random",
        env=env,
        n_episodes=args.n_episodes,
        rng=rng,
        device=device,
    )
    planner_stats = run_rollouts(
        model=model,
        policy_mode="planner",
        env=env,
        n_episodes=args.n_episodes,
        rng=rng,
        device=device,
    )

    random_metrics = coverage_metrics(random_stats, env=env, bins=args.bins)
    planner_metrics = coverage_metrics(planner_stats, env=env, bins=args.bins)
    arch = architecture_summary(model=model, cfg=cfg)
    comparison = make_comparison(arch)

    out_dir = PROJECT_ROOT / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.out_prefix:
        prefix = args.out_prefix
    else:
        prefix = f"failure_diag_{stamp}"
    out_json = out_dir / f"{prefix}.json"
    out_png = out_dir / f"{prefix}.png"

    payload = {
        "timestamp": stamp,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "n_episodes": int(args.n_episodes),
        "episode_len": episode_len,
        "bins": int(args.bins),
        "seed": int(args.seed),
        "current_architecture": arch,
        "maze2d_reference": MAZE2D_REFERENCE,
        "architecture_comparison": comparison,
        "random_policy_metrics": random_metrics,
        "planner_policy_metrics": planner_metrics,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_heatmaps(
        random_stats=random_stats,
        planner_stats=planner_stats,
        random_metrics=random_metrics,
        planner_metrics=planner_metrics,
        env=env,
        bins=args.bins,
        out_png=out_png,
    )

    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
