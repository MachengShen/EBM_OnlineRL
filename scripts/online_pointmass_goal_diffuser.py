#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
    total_env_steps: int = 30000
    warmup_steps: int = 5000
    train_every: int = 500
    gradient_steps: int = 200
    batch_size: int = 64
    horizon: int = 32
    n_diffusion_steps: int = 16
    predict_epsilon: bool = True
    clip_denoised: bool = False
    loss_action_weight: float = 10.0
    loss_discount: float = 1.0
    loss_obs_weight: float = 1.0
    holdout_episodes: int = 4
    val_every: int = 500
    val_batches: int = 8
    val_batch_size: int = 64
    eval_every: int = 10000
    n_eval_episodes: int = 20
    eval_goal_mode: str = "random"
    goal_sampling_p_replay: float = 0.5
    check_conditioning: bool = False
    episode_len: int = 50
    learning_rate: float = 3e-4
    max_episodes_in_replay: int = 10000
    model_base_dim: int = 32
    model_dim_mults: str = "1,2,4"
    use_ema_for_planning: bool = True
    ema_decay: float = 0.995
    ema_start_step: int = 0
    ema_update_every: int = 1
    planner_control_mode: str = "waypoint"
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
    parser.add_argument(
        "--predict_epsilon",
        dest="predict_epsilon",
        action="store_true",
        help="Train diffusion model to predict epsilon noise (default).",
    )
    parser.add_argument(
        "--predict_x0",
        dest="predict_epsilon",
        action="store_false",
        help="Train diffusion model to directly predict x0 (Diffuser Maze2D style).",
    )
    parser.set_defaults(predict_epsilon=TrainConfig.predict_epsilon)
    parser.add_argument(
        "--clip_denoised",
        dest="clip_denoised",
        action="store_true",
        help="Clamp reconstructed trajectories to [-1, 1] during sampling.",
    )
    parser.add_argument(
        "--no_clip_denoised",
        dest="clip_denoised",
        action="store_false",
        help="Disable trajectory clipping during diffusion sampling (default).",
    )
    parser.set_defaults(clip_denoised=TrainConfig.clip_denoised)
    parser.add_argument("--loss_action_weight", type=float, default=TrainConfig.loss_action_weight)
    parser.add_argument("--loss_discount", type=float, default=TrainConfig.loss_discount)
    parser.add_argument("--loss_obs_weight", type=float, default=TrainConfig.loss_obs_weight)
    parser.add_argument("--holdout_episodes", type=int, default=TrainConfig.holdout_episodes)
    parser.add_argument("--val_every", type=int, default=TrainConfig.val_every)
    parser.add_argument("--val_batches", type=int, default=TrainConfig.val_batches)
    parser.add_argument("--val_batch_size", type=int, default=TrainConfig.val_batch_size)
    parser.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--n_eval_episodes", type=int, default=TrainConfig.n_eval_episodes)
    parser.add_argument(
        "--eval_goal_mode",
        type=str,
        default=TrainConfig.eval_goal_mode,
        choices=("random", "reachable_from_random_trajectory"),
        help=(
            "Evaluation goal protocol. "
            "'random' samples independent start/goal from env; "
            "'reachable_from_random_trajectory' uses start and final state from the same random rollout."
        ),
    )
    parser.add_argument("--goal_sampling_p_replay", type=float, default=TrainConfig.goal_sampling_p_replay)
    parser.add_argument("--check_conditioning", action="store_true", default=TrainConfig.check_conditioning)
    parser.add_argument("--episode_len", type=int, default=TrainConfig.episode_len)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--max_episodes_in_replay", type=int, default=TrainConfig.max_episodes_in_replay)
    parser.add_argument("--model_base_dim", type=int, default=TrainConfig.model_base_dim)
    parser.add_argument("--model_dim_mults", type=str, default=TrainConfig.model_dim_mults)
    parser.add_argument(
        "--use_ema_for_planning",
        dest="use_ema_for_planning",
        action="store_true",
        help="Use EMA-smoothed diffusion model for planner/eval sampling (recommended).",
    )
    parser.add_argument(
        "--no_use_ema_for_planning",
        dest="use_ema_for_planning",
        action="store_false",
        help="Use raw online model for planner/eval sampling.",
    )
    parser.set_defaults(use_ema_for_planning=TrainConfig.use_ema_for_planning)
    parser.add_argument("--ema_decay", type=float, default=TrainConfig.ema_decay)
    parser.add_argument("--ema_start_step", type=int, default=TrainConfig.ema_start_step)
    parser.add_argument("--ema_update_every", type=int, default=TrainConfig.ema_update_every)
    parser.add_argument(
        "--planner_control_mode",
        type=str,
        default=TrainConfig.planner_control_mode,
        choices=("action", "waypoint"),
        help="Action extraction mode: direct predicted action or waypoint-based state tracking.",
    )
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


@torch.no_grad()
def hard_update_model(target_model: torch.nn.Module, source_model: torch.nn.Module) -> None:
    target_model.load_state_dict(source_model.state_dict())


@torch.no_grad()
def ema_update_model(target_model: torch.nn.Module, source_model: torch.nn.Module, decay: float) -> None:
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
    for target_buf, source_buf in zip(target_model.buffers(), source_model.buffers()):
        target_buf.copy_(source_buf)


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
    check_conditioning: bool,
    planner_control_mode: str,
    start_state: np.ndarray | None = None,
) -> Dict[str, np.ndarray | float]:
    obs = env.reset(goal=goal)
    if start_state is not None:
        env.state = np.asarray(start_state, dtype=np.float32).copy()
        obs = env.state.copy()
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
                check_conditioning=check_conditioning,
                control_mode=planner_control_mode,
            )
        else:
            raise ValueError(f"Unknown policy mode: {policy_mode}")

        obs, _, done, info = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())
        d = float(info["dist_to_goal"])
        min_dist = min(min_dist, d)
        final_dist = d

    success = float(min_dist <= env.success_threshold)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "final_dist": final_dist,
        "min_dist": min_dist,
        "success": success,
    }


def sample_reachable_start_goal_from_random_trajectory(
    env: PointMass2D,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a reachable (start, goal) pair from one random trajectory.

    This keeps evaluation in-distribution of the environment dynamics without
    requiring retraining.
    """
    warm_goal = env.sample_goal(rng)
    obs = env.reset(goal=warm_goal)
    states: List[np.ndarray] = [obs.copy()]
    done = False
    while not done:
        action = rng.uniform(-env.action_limit, env.action_limit, size=(env.act_dim,)).astype(np.float32)
        obs, _, done, _ = env.step(action)
        states.append(obs.copy())

    start_state = states[0].copy()
    goal_state = states[-1].copy()
    return start_state, goal_state


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
    ema_model: GaussianDiffusion1D | None = None,
    ema_decay: float = 0.995,
    ema_start_step: int = 0,
    ema_update_every: int = 1,
    update_step_offset: int = 0,
) -> tuple[float, int]:
    losses: List[float] = []
    model.train()
    updates_done = 0
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
        updates_done += 1
        if ema_model is not None:
            update_step = update_step_offset + updates_done
            if update_step < ema_start_step:
                hard_update_model(ema_model, model)
            elif update_step % max(1, ema_update_every) == 0:
                ema_update_model(ema_model, model, decay=ema_decay)
        losses.append(float(loss.item()))
    return (float(np.mean(losses)) if losses else float("nan"), updates_done)


@torch.no_grad()
def evaluate(
    model: GaussianDiffusion1D,
    n_episodes: int,
    seed: int,
    episode_len: int,
    eval_goal_mode: str,
    check_conditioning: bool,
    planner_control_mode: str,
    device: torch.device,
) -> Dict[str, float]:
    eval_env = PointMass2D(episode_length=episode_len)
    rng = np.random.default_rng(seed)
    model.eval()

    finals: List[float] = []
    mins: List[float] = []
    successes: List[float] = []
    for _ in range(n_episodes):
        if eval_goal_mode == "random":
            goal = eval_env.sample_goal(rng)
            start_state = None
        elif eval_goal_mode == "reachable_from_random_trajectory":
            start_state, goal = sample_reachable_start_goal_from_random_trajectory(eval_env, rng)
        else:
            raise ValueError(f"Unknown eval_goal_mode: {eval_goal_mode}")
        ep = rollout_episode(
            env=eval_env,
            goal=goal,
            policy_mode="planner",
            rng=rng,
            model=model,
            device=device,
            check_conditioning=check_conditioning,
            planner_control_mode=planner_control_mode,
            start_state=start_state,
        )
        finals.append(float(ep["final_dist"]))
        mins.append(float(ep["min_dist"]))
        successes.append(float(ep["success"]))

    return {
        "eval_success_rate": float(np.mean(successes)),
        "eval_final_dist_mean": float(np.mean(finals)),
        "eval_min_dist_mean": float(np.mean(mins)),
    }


@torch.no_grad()
def compute_validation_loss(
    model: GaussianDiffusion1D,
    replay: EpisodeReplayBuffer,
    rng: np.random.Generator,
    batch_size: int,
    horizon: int,
    n_batches: int,
    action_scale: float,
    device: torch.device,
) -> float:
    if n_batches <= 0 or not replay.can_sample(horizon):
        return float("nan")

    losses: List[float] = []
    model.eval()
    for _ in range(n_batches):
        batch_np = replay.sample_trajectory_segment(
            batch_size=batch_size,
            horizon=horizon,
            rng=rng,
            action_scale=action_scale,
        )
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        loss = model.loss(batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable but a CUDA device was requested. "
                "Install CUDA-enabled PyTorch and check NVIDIA driver visibility."
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print(
            "Warning: running on CPU. This will be slow due to diffusion MPC planning.",
            flush=True,
        )

    logdir = make_logdir(cfg)
    with (logdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    env = PointMass2D(episode_length=cfg.episode_len)
    replay = EpisodeReplayBuffer(obs_dim=env.obs_dim, act_dim=env.act_dim, max_episodes=cfg.max_episodes_in_replay)
    val_replay = EpisodeReplayBuffer(obs_dim=env.obs_dim, act_dim=env.act_dim, max_episodes=max(cfg.holdout_episodes, 1))

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
        action_dim=env.act_dim,
        n_diffusion_steps=cfg.n_diffusion_steps,
        predict_epsilon=cfg.predict_epsilon,
        clip_denoised=cfg.clip_denoised,
        action_weight=cfg.loss_action_weight,
        loss_discount=cfg.loss_discount,
        obs_loss_weight=cfg.loss_obs_weight,
    ).to(device)
    ema_diffusion = copy.deepcopy(diffusion).to(device)
    hard_update_model(ema_diffusion, diffusion)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.learning_rate)
    metrics_path = logdir / "metrics.jsonl"

    env_steps = 0
    next_train = cfg.train_every
    next_val = cfg.val_every
    next_eval = cfg.eval_every

    eval_history_steps: List[int] = []
    eval_history_success: List[float] = []
    train_updates = 0

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
            check_conditioning=cfg.check_conditioning,
            planner_control_mode=cfg.planner_control_mode,
        )
        if len(val_replay) < cfg.holdout_episodes:
            val_replay.add_episode(ep["states"], ep["actions"])
        else:
            replay.add_episode(ep["states"], ep["actions"])
        env_steps += int(ep["actions"].shape[0])

    print(
        f"Warmup done: env_steps={env_steps}, train_episodes={len(replay)}, "
        f"holdout_episodes={len(val_replay)}, train_can_sample={replay.can_sample(cfg.horizon)}, "
        f"holdout_can_sample={val_replay.can_sample(cfg.horizon)}",
        flush=True,
    )
    burnin_loss = float("nan")
    if replay.can_sample(cfg.horizon):
        burnin_loss, burnin_updates = train_burst(
            model=diffusion,
            optimizer=optimizer,
            replay=replay,
            rng=rng,
            batch_size=cfg.batch_size,
            horizon=cfg.horizon,
            gradient_steps=cfg.gradient_steps,
            action_scale=env.action_limit,
            device=device,
            ema_model=ema_diffusion if cfg.use_ema_for_planning else None,
            ema_decay=cfg.ema_decay,
            ema_start_step=cfg.ema_start_step,
            ema_update_every=cfg.ema_update_every,
            update_step_offset=train_updates,
        )
        train_updates += burnin_updates
    print(f"Burn-in training done: loss={burnin_loss}", flush=True)
    next_train = env_steps + cfg.train_every

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
            model=ema_diffusion if cfg.use_ema_for_planning else diffusion,
            device=device,
            check_conditioning=cfg.check_conditioning,
            planner_control_mode=cfg.planner_control_mode,
        )
        replay.add_episode(ep["states"], ep["actions"])
        env_steps += int(ep["actions"].shape[0])

        train_loss = float("nan")
        if env_steps >= next_train and replay.can_sample(cfg.horizon):
            train_loss, n_updates = train_burst(
                model=diffusion,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
                batch_size=cfg.batch_size,
                horizon=cfg.horizon,
                gradient_steps=cfg.gradient_steps,
                action_scale=env.action_limit,
                device=device,
                ema_model=ema_diffusion if cfg.use_ema_for_planning else None,
                ema_decay=cfg.ema_decay,
                ema_start_step=cfg.ema_start_step,
                ema_update_every=cfg.ema_update_every,
                update_step_offset=train_updates,
            )
            train_updates += n_updates
            next_train += cfg.train_every

        val_loss = float("nan")
        if cfg.val_every > 0 and env_steps >= next_val and val_replay.can_sample(cfg.horizon):
            val_loss = compute_validation_loss(
                model=diffusion,
                replay=val_replay,
                rng=rng,
                batch_size=cfg.val_batch_size,
                horizon=cfg.horizon,
                n_batches=cfg.val_batches,
                action_scale=env.action_limit,
                device=device,
            )
            next_val += cfg.val_every

        eval_payload: Dict[str, float] = {}
        if env_steps >= next_eval:
            eval_payload = evaluate(
                model=ema_diffusion if cfg.use_ema_for_planning else diffusion,
                n_episodes=cfg.n_eval_episodes,
                seed=cfg.seed + env_steps + 13,
                episode_len=cfg.episode_len,
                eval_goal_mode=cfg.eval_goal_mode,
                check_conditioning=cfg.check_conditioning,
                planner_control_mode=cfg.planner_control_mode,
                device=device,
            )
            eval_history_steps.append(env_steps)
            eval_history_success.append(eval_payload["eval_success_rate"])

            ckpt = {
                "env_steps": env_steps,
                "model_state_dict": diffusion.state_dict(),
                "ema_model_state_dict": ema_diffusion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
            }
            torch.save(ckpt, logdir / "checkpoints" / f"step_{env_steps}.pt")
            next_eval += cfg.eval_every

        train_val_gap = float("nan")
        if np.isfinite(train_loss) and np.isfinite(val_loss):
            train_val_gap = float(val_loss - train_loss)

        payload: Dict[str, float | int] = {
            "env_steps": env_steps,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_val_gap": train_val_gap,
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
