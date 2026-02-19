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

from ebm_online_rl.online import GaussianDiffusion1D, TemporalUNet1D


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cuda:0"
    logdir: str = ""
    horizon: int = 32
    n_diffusion_steps: int = 16
    predict_epsilon: bool = False
    clip_denoised: bool = False
    model_base_dim: int = 32
    model_dim_mults: str = "1,2,4"
    learning_rate: float = 3e-4
    train_steps: int = 1200
    batch_size: int = 128
    grad_clip: float = 1.0
    val_every: int = 20
    val_batches: int = 8
    log_every: int = 50
    n_train_traj: int = 6000
    n_val_traj: int = 1200
    start_min: float = -1.0
    start_max: float = 1.0
    step_min: float = 0.02
    step_max: float = 0.08
    action_scale: float = 0.1
    diag_start: str = "-0.8,-0.8"
    diag_goal: str = "0.8,0.8"
    n_diag_samples: int = 16


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Offline diffusion experiment on a synthetic dataset with only horizontal/vertical trajectories, "
            "plus diagonal start-goal generalization check."
        )
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--device", type=str, default=ExperimentConfig.device)
    parser.add_argument("--logdir", type=str, default=ExperimentConfig.logdir)
    parser.add_argument("--horizon", type=int, default=ExperimentConfig.horizon)
    parser.add_argument("--n_diffusion_steps", type=int, default=ExperimentConfig.n_diffusion_steps)
    parser.add_argument(
        "--predict_epsilon",
        dest="predict_epsilon",
        action="store_true",
        help="Train model to predict epsilon noise.",
    )
    parser.add_argument(
        "--predict_x0",
        dest="predict_epsilon",
        action="store_false",
        help="Train model to predict x0 directly.",
    )
    parser.set_defaults(predict_epsilon=ExperimentConfig.predict_epsilon)
    parser.add_argument(
        "--clip_denoised",
        dest="clip_denoised",
        action="store_true",
        help="Clamp denoised trajectory values to [-1, 1] during sampling.",
    )
    parser.add_argument(
        "--no_clip_denoised",
        dest="clip_denoised",
        action="store_false",
        help="Disable denoised trajectory clipping during sampling (default).",
    )
    parser.set_defaults(clip_denoised=ExperimentConfig.clip_denoised)
    parser.add_argument("--model_base_dim", type=int, default=ExperimentConfig.model_base_dim)
    parser.add_argument("--model_dim_mults", type=str, default=ExperimentConfig.model_dim_mults)
    parser.add_argument("--learning_rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--train_steps", type=int, default=ExperimentConfig.train_steps)
    parser.add_argument("--batch_size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--grad_clip", type=float, default=ExperimentConfig.grad_clip)
    parser.add_argument("--val_every", type=int, default=ExperimentConfig.val_every)
    parser.add_argument("--val_batches", type=int, default=ExperimentConfig.val_batches)
    parser.add_argument("--log_every", type=int, default=ExperimentConfig.log_every)
    parser.add_argument("--n_train_traj", type=int, default=ExperimentConfig.n_train_traj)
    parser.add_argument("--n_val_traj", type=int, default=ExperimentConfig.n_val_traj)
    parser.add_argument("--start_min", type=float, default=ExperimentConfig.start_min)
    parser.add_argument("--start_max", type=float, default=ExperimentConfig.start_max)
    parser.add_argument("--step_min", type=float, default=ExperimentConfig.step_min)
    parser.add_argument("--step_max", type=float, default=ExperimentConfig.step_max)
    parser.add_argument("--action_scale", type=float, default=ExperimentConfig.action_scale)
    parser.add_argument("--diag_start", type=str, default=ExperimentConfig.diag_start)
    parser.add_argument("--diag_goal", type=str, default=ExperimentConfig.diag_goal)
    parser.add_argument("--n_diag_samples", type=int, default=ExperimentConfig.n_diag_samples)
    return ExperimentConfig(**vars(parser.parse_args()))


def parse_vec2(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y', got: {value}")
    return np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32)


def parse_dim_mults(value: str) -> tuple[int, ...]:
    dim_mults = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not dim_mults:
        raise ValueError("--model_dim_mults cannot be empty")
    return dim_mults


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logdir(cfg: ExperimentConfig) -> Path:
    if cfg.logdir:
        out = Path(cfg.logdir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / "offline_hv_diffuser" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_hv_dataset(
    n_traj: int,
    horizon: int,
    action_scale: float,
    start_min: float,
    start_max: float,
    step_min: float,
    step_max: float,
    seed: int,
) -> tuple[np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    transition_dim = 4
    directions = np.asarray(
        [
            [1.0, 0.0],   # left->right
            [-1.0, 0.0],  # right->left
            [0.0, -1.0],  # up->down
            [0.0, 1.0],   # down->up
        ],
        dtype=np.float32,
    )
    dir_names = [
        "left_to_right",
        "right_to_left",
        "up_to_down",
        "down_to_up",
    ]

    dir_idx = rng.integers(0, 4, size=n_traj)
    step_size = rng.uniform(step_min, step_max, size=(n_traj, 1)).astype(np.float32)
    action = directions[dir_idx] * step_size
    start = rng.uniform(start_min, start_max, size=(n_traj, 2)).astype(np.float32)

    t = np.arange(horizon + 1, dtype=np.float32)[None, :, None]
    states = start[:, None, :] + t * action[:, None, :]  # [N, H+1, 2]
    actions = np.repeat(action[:, None, :], horizon, axis=1).astype(np.float32)  # [N, H, 2]

    packed = np.zeros((n_traj, horizon + 1, transition_dim), dtype=np.float32)
    packed[:, :-1, :2] = states[:, :-1, :]
    packed[:, :-1, 2:] = actions / action_scale
    packed[:, -1, :2] = states[:, -1, :]

    direction_hist = {dir_names[i]: int(np.sum(dir_idx == i)) for i in range(4)}
    return packed, direction_hist


def sample_batch(data: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, data.shape[0], size=batch_size)
    return data[idx]


@torch.no_grad()
def compute_val_loss(
    model: GaussianDiffusion1D,
    val_data: np.ndarray,
    batch_size: int,
    n_batches: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(n_batches):
        batch_np = sample_batch(val_data, batch_size=batch_size, rng=rng)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        loss = model.loss(batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def plot_losses(metrics: List[Dict[str, float]], out_path: Path) -> None:
    steps = [m["step"] for m in metrics]
    train_losses = [m["train_loss"] for m in metrics]
    val_steps = [m["step"] for m in metrics if np.isfinite(m.get("val_loss", np.nan))]
    val_losses = [m["val_loss"] for m in metrics if np.isfinite(m.get("val_loss", np.nan))]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_losses, label="train_loss", alpha=0.7)
    if val_steps:
        plt.plot(val_steps, val_losses, marker="o", label="val_loss")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.title("HV Synthetic Dataset: Train/Val Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@torch.no_grad()
def evaluate_diagonal_generalization(
    model: GaussianDiffusion1D,
    start: np.ndarray,
    goal: np.ndarray,
    n_samples: int,
    action_scale: float,
    device: torch.device,
) -> Dict:
    model.eval()
    obs0_t = torch.as_tensor(start, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    traj = model.sample(batch_size=n_samples, obs0=obs0_t, goal=goal_t, obs_dim=2, act_dim=2)
    traj_np = traj.detach().cpu().numpy()

    states = traj_np[:, :, :2]
    actions = traj_np[:, :-1, 2:4] * action_scale
    horizon = states.shape[1] - 1

    alpha = np.linspace(0.0, 1.0, horizon + 1, dtype=np.float32)
    line = start[None, :] * (1.0 - alpha[:, None]) + goal[None, :] * alpha[:, None]  # [H+1,2]
    line_dev = np.linalg.norm(states - line[None, :, :], axis=-1).mean(axis=1)
    final_dist = np.linalg.norm(states[:, -1, :] - goal[None, :], axis=-1)
    first_action = actions[:, 0, :]

    return {
        "states": states,
        "actions": actions,
        "metrics": {
            "line_dev_mean": float(np.mean(line_dev)),
            "line_dev_std": float(np.std(line_dev)),
            "final_dist_mean": float(np.mean(final_dist)),
            "final_dist_std": float(np.std(final_dist)),
            "first_action_mean": first_action.mean(axis=0).tolist(),
            "first_action_std": first_action.std(axis=0).tolist(),
        },
    }


def plot_diagonal_trajectories(
    states: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    for i in range(states.shape[0]):
        plt.plot(states[i, :, 0], states[i, :, 1], alpha=0.5)
    plt.plot([start[0], goal[0]], [start[1], goal[1]], "k--", linewidth=2, label="target line")
    plt.scatter([start[0]], [start[1]], c="green", s=60, label="start")
    plt.scatter([goal[0]], [goal[1]], c="red", s=60, label="goal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Diagonal Start/Goal Generalization")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print("Warning: running on CPU; this may be slower.", flush=True)

    logdir = make_logdir(cfg)
    with (logdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_data, train_hist = build_hv_dataset(
        n_traj=cfg.n_train_traj,
        horizon=cfg.horizon,
        action_scale=cfg.action_scale,
        start_min=cfg.start_min,
        start_max=cfg.start_max,
        step_min=cfg.step_min,
        step_max=cfg.step_max,
        seed=cfg.seed + 1,
    )
    val_data, val_hist = build_hv_dataset(
        n_traj=cfg.n_val_traj,
        horizon=cfg.horizon,
        action_scale=cfg.action_scale,
        start_min=cfg.start_min,
        start_max=cfg.start_max,
        step_min=cfg.step_min,
        step_max=cfg.step_max,
        seed=cfg.seed + 2,
    )

    dim_mults = parse_dim_mults(cfg.model_dim_mults)
    denoiser = TemporalUNet1D(transition_dim=4, base_dim=cfg.model_base_dim, dim_mults=dim_mults)
    model = GaussianDiffusion1D(
        model=denoiser,
        horizon=cfg.horizon,
        transition_dim=4,
        action_dim=2,
        n_diffusion_steps=cfg.n_diffusion_steps,
        predict_epsilon=cfg.predict_epsilon,
        clip_denoised=cfg.clip_denoised,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    metrics: List[Dict[str, float]] = []
    metrics_path = logdir / "metrics.jsonl"
    for step in range(1, cfg.train_steps + 1):
        model.train()
        batch_np = sample_batch(train_data, batch_size=cfg.batch_size, rng=rng)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        entry: Dict[str, float] = {
            "step": float(step),
            "train_loss": float(loss.item()),
            "val_loss": float("nan"),
        }
        if step == 1 or step % cfg.val_every == 0:
            entry["val_loss"] = compute_val_loss(
                model=model,
                val_data=val_data,
                batch_size=cfg.batch_size,
                n_batches=cfg.val_batches,
                rng=rng,
                device=device,
            )

        metrics.append(entry)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if step == 1 or step % cfg.log_every == 0 or np.isfinite(entry["val_loss"]):
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": entry["train_loss"],
                        "val_loss": entry["val_loss"],
                    }
                ),
                flush=True,
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "config": asdict(cfg)},
        logdir / "model_last.pt",
    )

    plot_losses(metrics, out_path=logdir / "loss_curve.png")

    start = parse_vec2(cfg.diag_start)
    goal = parse_vec2(cfg.diag_goal)
    diag = evaluate_diagonal_generalization(
        model=model,
        start=start,
        goal=goal,
        n_samples=cfg.n_diag_samples,
        action_scale=cfg.action_scale,
        device=device,
    )
    plot_diagonal_trajectories(diag["states"], start=start, goal=goal, out_path=logdir / "diag_generalization.png")

    val_points = [m for m in metrics if np.isfinite(m["val_loss"])]
    train_first = metrics[0]["train_loss"]
    train_last = metrics[-1]["train_loss"]
    val_first = val_points[0]["val_loss"] if val_points else float("nan")
    val_last = val_points[-1]["val_loss"] if val_points else float("nan")

    summary = {
        "logdir": str(logdir),
        "train_dataset_direction_hist": train_hist,
        "val_dataset_direction_hist": val_hist,
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "val_loss_first": val_first,
        "val_loss_last": val_last,
        "diag_eval": diag["metrics"],
    }
    with (logdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment complete.")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts saved under: {logdir}")


if __name__ == "__main__":
    main()
