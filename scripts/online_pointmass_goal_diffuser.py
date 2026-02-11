#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ebm_online_rl.envs import PointMass2D
from ebm_online_rl.online import EpisodeReplayBuffer, GaussianDiffusion1D, TemporalUNet1D, plan_action


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda:0"
    total_env_steps: int = 20000
    warmup_steps: int = 5000
    train_every: int = 1000
    gradient_steps: int = 200
    batch_size: int = 64
    horizon: int = 32
    n_diffusion_steps: int = 32
    eval_every: int = 5000
    n_eval_episodes: int = 50
    goal_sampling_p_replay: float = 0.5
    n_plan_samples: int = 1
    episode_len: int = 50
    learning_rate: float = 3e-4
    max_episodes_in_replay: int = 10000
    model_base_dim: int = 64
    model_dim_mults: str = "1,2,4"
    logdir: str = ""


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Online PointMass goal-reaching with diffusion planning.")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--total_env_steps", type=int, default=TrainConfig.total_env_steps)
    parser.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--train_every", type=int, default=TrainConfig.train_every)
    parser.add_argument("--gradient_steps", type=int, default=TrainConfig.gradient_steps)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    parser.add_argument("--n_diffusion_steps", type=int, default=TrainConfig.n_diffusion_steps)
    parser.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--n_eval_episodes", type=int, default=TrainConfig.n_eval_episodes)
    parser.add_argument("--goal_sampling_p_replay", type=float, default=TrainConfig.goal_sampling_p_replay)
    parser.add_argument("--n_plan_samples", type=int, default=TrainConfig.n_plan_samples)
    parser.add_argument("--episode_len", type=int, default=TrainConfig.episode_len)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--max_episodes_in_replay", type=int, default=TrainConfig.max_episodes_in_replay)
    parser.add_argument("--model_base_dim", type=int, default=TrainConfig.model_base_dim)
    parser.add_argument("--model_dim_mults", type=str, default=TrainConfig.model_dim_mults)
    parser.add_argument("--logdir", type=str, default=TrainConfig.logdir)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def make_logdir(cfg: TrainConfig) -> Path:
    if cfg.logdir:
        out = Path(cfg.logdir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / "online_pointmass_goal_diffuser" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_goal(
    env: PointMass2D,
    replay: EpisodeReplayBuffer,
    rng: np.random.Generator,
    p_replay: float,
) -> np.ndarray:
    use_replay = len(replay) > 0 and rng.random() < p_replay
    if use_replay:
        return replay.sample_achieved_goal(rng)
    return env.sample_goal(rng)


def rollout_episode(
    env: PointMass2D,
    goal: np.ndarray,
    policy_mode: str,
    rng: np.random.Generator,
    model: GaussianDiffusion1D | None,
    device: torch.device,
    n_plan_samples: int,
) -> Dict[str, np.ndarray | float]:
    obs = env.reset(goal=goal)
    states: List[np.ndarray] = [obs.copy()]
    actions: List[np.ndarray] = []
    done = False
    min_dist = float(np.linalg.norm(obs - goal))
    final_dist = min_dist

    while not done:
        if policy_mode == "random":
            action = rng.uniform(-env.action_limit, env.action_limit, size=(env.act_dim,)).astype(np.float32)
        elif policy_mode == "planner":
            if model is None:
                raise ValueError("model is required for planner policy mode")
            action = plan_action(
                model=model,
                obs=obs,
                goal=goal,
                obs_dim=env.obs_dim,
                act_dim=env.act_dim,
                action_scale=env.action_limit,
                device=device,
                n_samples=n_plan_samples,
                check_conditioning=True,
            )
        else:
            raise ValueError(f"Unknown policy mode: {policy_mode}")

        obs, _, done, info = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())
        d = float(info["dist_to_goal"])
        min_dist = min(min_dist, d)
        final_dist = d

    success = float(final_dist <= env.success_threshold)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "final_dist": final_dist,
        "min_dist": min_dist,
        "success": success,
    }


def train_burst(
    model: GaussianDiffusion1D,
    optimizer: torch.optim.Optimizer,
    replay: EpisodeReplayBuffer,
    rng: np.random.Generator,
    batch_size: int,
    horizon: int,
    gradient_steps: int,
    action_scale: float,
    device: torch.device,
) -> float:
    losses: List[float] = []
    model.train()
    for _ in range(gradient_steps):
        batch_np = replay.sample_trajectory_segment(
            batch_size=batch_size,
            horizon=horizon,
            rng=rng,
            action_scale=action_scale,
        )
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    model: GaussianDiffusion1D,
    n_episodes: int,
    seed: int,
    episode_len: int,
    n_plan_samples: int,
    device: torch.device,
) -> Dict[str, float]:
    eval_env = PointMass2D(episode_length=episode_len)
    rng = np.random.default_rng(seed)
    model.eval()

    finals: List[float] = []
    mins: List[float] = []
    successes: List[float] = []
    for _ in range(n_episodes):
        goal = eval_env.sample_goal(rng)
        ep = rollout_episode(
            env=eval_env,
            goal=goal,
            policy_mode="planner",
            rng=rng,
            model=model,
            device=device,
            n_plan_samples=n_plan_samples,
        )
        finals.append(float(ep["final_dist"]))
        mins.append(float(ep["min_dist"]))
        successes.append(float(ep["success"]))

    return {
        "eval_success_rate": float(np.mean(successes)),
        "eval_final_dist_mean": float(np.mean(finals)),
        "eval_min_dist_mean": float(np.mean(mins)),
    }


def append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if not cfg.device.startswith("cuda"):
        raise ValueError(
            f"--device must be CUDA to keep training compute on GPU, got '{cfg.device}'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable but GPU training was requested. "
            "Install CUDA-enabled PyTorch and check NVIDIA driver visibility."
        )
    device = torch.device(cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logdir = make_logdir(cfg)
    with (logdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    env = PointMass2D(episode_length=cfg.episode_len)
    replay = EpisodeReplayBuffer(obs_dim=env.obs_dim, act_dim=env.act_dim, max_episodes=cfg.max_episodes_in_replay)

    transition_dim = env.obs_dim + env.act_dim
    dim_mults = tuple(int(x.strip()) for x in cfg.model_dim_mults.split(",") if x.strip())
    if not dim_mults:
        raise ValueError("--model_dim_mults cannot be empty.")
    denoiser = TemporalUNet1D(
        transition_dim=transition_dim,
        base_dim=cfg.model_base_dim,
        dim_mults=dim_mults,
    )
    diffusion = GaussianDiffusion1D(
        model=denoiser,
        horizon=cfg.horizon,
        transition_dim=transition_dim,
        n_diffusion_steps=cfg.n_diffusion_steps,
        predict_epsilon=True,
        clip_denoised=True,
    ).to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.learning_rate)
    metrics_path = logdir / "metrics.jsonl"

    env_steps = 0
    next_train = cfg.warmup_steps + cfg.train_every
    next_eval = cfg.eval_every

    eval_history_steps: List[int] = []
    eval_history_success: List[float] = []

    print("Starting warmup collection...", flush=True)
    while env_steps < cfg.warmup_steps:
        goal = env.sample_goal(rng)
        ep = rollout_episode(
            env=env,
            goal=goal,
            policy_mode="random",
            rng=rng,
            model=None,
            device=device,
            n_plan_samples=cfg.n_plan_samples,
        )
        replay.add_episode(ep["states"], ep["actions"])
        env_steps += int(ep["actions"].shape[0])

    print(
        f"Warmup done: env_steps={env_steps}, episodes={len(replay)}, can_sample={replay.can_sample(cfg.horizon)}",
        flush=True,
    )

    while env_steps < cfg.total_env_steps:
        goal = choose_goal(
            env=env,
            replay=replay,
            rng=rng,
            p_replay=cfg.goal_sampling_p_replay,
        )
        ep = rollout_episode(
            env=env,
            goal=goal,
            policy_mode="planner",
            rng=rng,
            model=diffusion,
            device=device,
            n_plan_samples=cfg.n_plan_samples,
        )
        replay.add_episode(ep["states"], ep["actions"])
        env_steps += int(ep["actions"].shape[0])

        train_loss = float("nan")
        if env_steps >= next_train and replay.can_sample(cfg.horizon):
            train_loss = train_burst(
                model=diffusion,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
                batch_size=cfg.batch_size,
                horizon=cfg.horizon,
                gradient_steps=cfg.gradient_steps,
                action_scale=env.action_limit,
                device=device,
            )
            next_train += cfg.train_every

        eval_payload: Dict[str, float] = {}
        if env_steps >= next_eval:
            eval_payload = evaluate(
                model=diffusion,
                n_episodes=cfg.n_eval_episodes,
                seed=cfg.seed + env_steps + 13,
                episode_len=cfg.episode_len,
                n_plan_samples=cfg.n_plan_samples,
                device=device,
            )
            eval_history_steps.append(env_steps)
            eval_history_success.append(eval_payload["eval_success_rate"])

            ckpt = {
                "env_steps": env_steps,
                "model_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
            }
            torch.save(ckpt, logdir / "checkpoints" / f"step_{env_steps}.pt")
            next_eval += cfg.eval_every

        payload: Dict[str, float | int] = {
            "env_steps": env_steps,
            "train_loss": train_loss,
            "episode_final_dist": float(ep["final_dist"]),
            "episode_min_dist": float(ep["min_dist"]),
            "episode_success": float(ep["success"]),
            "num_episodes_in_replay": len(replay),
        }
        payload.update(eval_payload)
        append_jsonl(metrics_path, payload)

        print(json.dumps(payload), flush=True)

    if eval_history_steps:
        plt.figure(figsize=(8, 4))
        plt.plot(eval_history_steps, eval_history_success, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Environment Steps")
        plt.ylabel("Eval Success Rate")
        plt.title("Online PointMass Goal-Reaching")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(logdir / "success_rate.png", dpi=150)
        plt.close()

    print(f"Finished training. Logs written to: {logdir}", flush=True)


if __name__ == "__main__":
    main()
