#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reuse Maze2D data/eval/plot helpers from the existing diffuser probe.
REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_BIN = "/root/.mujoco/mujoco210/bin"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if MUJOCO_BIN not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{_ld}:{MUJOCO_BIN}" if _ld else MUJOCO_BIN
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
sys.path.insert(0, str(REPO_ROOT / "third_party" / "diffuser-maze2d"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import synthetic_maze2d_diffuser_probe as dprobe  # noqa: E402


@dataclass
class Config:
    env: str = "maze2d-umaze-v1"
    seed: int = 0
    device: str = "cuda:0"
    logdir: str = ""
    n_episodes: int = 400
    episode_len: int = 192
    horizon: int = 64
    max_path_length: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    train_steps: int = 4000
    batch_size: int = 256
    grad_clip: float = 1.0
    val_frac: float = 0.1
    val_every: int = 100
    val_batches: int = 20
    gcbc_hidden_dims: str = "256,256"
    gcbc_her_k_per_transition: int = 4
    gcbc_future_sample_attempts: int = 16
    action_scale: float = 1.0
    corridor_aware_data: bool = False
    corridor_max_resamples: int = 200
    query: str = "0.9,2.9:2.9,2.9;0.9,2.9:2.9,0.9;2.9,0.9:0.9,2.9"
    query_mode: str = "diverse"
    num_eval_queries: int = 24
    query_bank_size: int = 256
    query_angle_bins: int = 16
    query_min_distance: float = 1.0
    query_resample_each_eval: bool = True
    query_resample_seed_stride: int = 7919
    query_batch_size: int = 6
    eval_goal_every: int = 5000
    goal_success_threshold: float = 0.5
    eval_rollout_mode: str = "receding_horizon"
    eval_rollout_replan_every_n_steps: int = 8
    eval_rollout_horizon: int = 256
    eval_success_prefix_horizons: str = "64,128,192,256"
    save_checkpoint_every: int = 5000
    online_self_improve: bool = False
    online_rounds: int = 0
    online_train_steps_per_round: int = 2000
    online_collect_episodes_per_round: int = 32
    online_collect_episode_len: int = 256
    online_collect_transition_budget_per_round: int = 0
    online_replan_every_n_steps: int = 8
    online_goal_geom_p: float = 0.08
    online_goal_geom_min_k: int = 8
    online_goal_geom_max_k: int = 96
    online_goal_min_distance: float = 0.5
    online_planning_success_thresholds: str = "0.1,0.2"
    online_planning_success_rel_reduction: float = 0.9
    online_early_terminate_on_success: bool = True
    online_early_terminate_threshold: float = 0.2
    online_min_accepted_episode_len: int = 0


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=(
            "Synthetic Maze2D experiment: Goal-Conditioned Behavior Cloning + "
            "Hindsight Experience Replay (HER), aligned with diffuser probe eval/online protocol."
        )
    )
    p.add_argument("--env", type=str, default=Config.env)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--logdir", type=str, default=Config.logdir)
    p.add_argument("--n_episodes", type=int, default=Config.n_episodes)
    p.add_argument("--episode_len", type=int, default=Config.episode_len)
    p.add_argument("--horizon", type=int, default=Config.horizon)
    p.add_argument("--max_path_length", type=int, default=Config.max_path_length)
    p.add_argument("--learning_rate", type=float, default=Config.learning_rate)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--train_steps", type=int, default=Config.train_steps)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--grad_clip", type=float, default=Config.grad_clip)
    p.add_argument("--val_frac", type=float, default=Config.val_frac)
    p.add_argument("--val_every", type=int, default=Config.val_every)
    p.add_argument("--val_batches", type=int, default=Config.val_batches)
    p.add_argument("--gcbc_hidden_dims", type=str, default=Config.gcbc_hidden_dims)
    p.add_argument("--gcbc_her_k_per_transition", type=int, default=Config.gcbc_her_k_per_transition)
    p.add_argument("--gcbc_future_sample_attempts", type=int, default=Config.gcbc_future_sample_attempts)
    p.add_argument("--action_scale", type=float, default=Config.action_scale)
    p.add_argument(
        "--corridor_aware_data",
        dest="corridor_aware_data",
        action="store_true",
        help="Reject synthetic random-policy transitions that land in wall cells.",
    )
    p.add_argument(
        "--no_corridor_aware_data",
        dest="corridor_aware_data",
        action="store_false",
    )
    p.set_defaults(corridor_aware_data=Config.corridor_aware_data)
    p.add_argument("--corridor_max_resamples", type=int, default=Config.corridor_max_resamples)
    p.add_argument("--query", type=str, default=Config.query)
    p.add_argument("--query_mode", type=str, choices=["fixed", "diverse"], default=Config.query_mode)
    p.add_argument("--num_eval_queries", type=int, default=Config.num_eval_queries)
    p.add_argument("--query_bank_size", type=int, default=Config.query_bank_size)
    p.add_argument("--query_angle_bins", type=int, default=Config.query_angle_bins)
    p.add_argument("--query_min_distance", type=float, default=Config.query_min_distance)
    p.add_argument(
        "--query_resample_each_eval",
        dest="query_resample_each_eval",
        action="store_true",
        help="For diverse query mode, resample query subset each eval step.",
    )
    p.add_argument(
        "--no_query_resample_each_eval",
        dest="query_resample_each_eval",
        action="store_false",
    )
    p.set_defaults(query_resample_each_eval=Config.query_resample_each_eval)
    p.add_argument("--query_resample_seed_stride", type=int, default=Config.query_resample_seed_stride)
    p.add_argument("--query_batch_size", type=int, default=Config.query_batch_size)
    p.add_argument("--eval_goal_every", type=int, default=Config.eval_goal_every)
    p.add_argument("--goal_success_threshold", type=float, default=Config.goal_success_threshold)
    p.add_argument(
        "--eval_rollout_mode",
        type=str,
        choices=["open_loop", "receding_horizon"],
        default=Config.eval_rollout_mode,
    )
    p.add_argument(
        "--eval_rollout_replan_every_n_steps",
        type=int,
        default=Config.eval_rollout_replan_every_n_steps,
        help="For GCBC this is decision cadence (action refresh stride).",
    )
    p.add_argument("--eval_rollout_horizon", type=int, default=Config.eval_rollout_horizon)
    p.add_argument("--eval_success_prefix_horizons", type=str, default=Config.eval_success_prefix_horizons)
    p.add_argument("--save_checkpoint_every", type=int, default=Config.save_checkpoint_every)

    p.add_argument(
        "--online_self_improve",
        dest="online_self_improve",
        action="store_true",
        help="Enable online rounds: collect real rollouts and continue GCBC+HER training.",
    )
    p.add_argument("--no_online_self_improve", dest="online_self_improve", action="store_false")
    p.set_defaults(online_self_improve=Config.online_self_improve)
    p.add_argument("--online_rounds", type=int, default=Config.online_rounds)
    p.add_argument("--online_train_steps_per_round", type=int, default=Config.online_train_steps_per_round)
    p.add_argument("--online_collect_episodes_per_round", type=int, default=Config.online_collect_episodes_per_round)
    p.add_argument("--online_collect_episode_len", type=int, default=Config.online_collect_episode_len)
    p.add_argument("--online_collect_transition_budget_per_round", type=int, default=Config.online_collect_transition_budget_per_round)
    p.add_argument(
        "--online_replan_every_n_steps",
        type=int,
        default=Config.online_replan_every_n_steps,
        help="For GCBC this is decision cadence during online collection.",
    )
    p.add_argument("--online_goal_geom_p", type=float, default=Config.online_goal_geom_p)
    p.add_argument("--online_goal_geom_min_k", type=int, default=Config.online_goal_geom_min_k)
    p.add_argument("--online_goal_geom_max_k", type=int, default=Config.online_goal_geom_max_k)
    p.add_argument("--online_goal_min_distance", type=float, default=Config.online_goal_min_distance)
    p.add_argument("--online_planning_success_thresholds", type=str, default=Config.online_planning_success_thresholds)
    p.add_argument("--online_planning_success_rel_reduction", type=float, default=Config.online_planning_success_rel_reduction)
    p.add_argument(
        "--online_early_terminate_on_success",
        dest="online_early_terminate_on_success",
        action="store_true",
    )
    p.add_argument(
        "--no_online_early_terminate_on_success",
        dest="online_early_terminate_on_success",
        action="store_false",
    )
    p.set_defaults(online_early_terminate_on_success=Config.online_early_terminate_on_success)
    p.add_argument("--online_early_terminate_threshold", type=float, default=Config.online_early_terminate_threshold)
    p.add_argument("--online_min_accepted_episode_len", type=int, default=Config.online_min_accepted_episode_len)
    return Config(**vars(p.parse_args()))


def parse_hidden_dims(raw: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if len(vals) == 0 or any(v <= 0 for v in vals):
        raise ValueError("--gcbc_hidden_dims must contain positive integers")
    return vals


class GCBCHERDataset(torch.utils.data.Dataset):
    def __init__(self, observations: np.ndarray, goals: np.ndarray, actions: np.ndarray):
        if len(observations) != len(goals) or len(goals) != len(actions):
            raise ValueError("GCBCHERDataset arrays must have equal length")
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.goals = torch.as_tensor(goals, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.goals[idx], self.actions[idx]


class GCBCPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        in_dim = int(observation_dim + goal_dim)
        dims = [in_dim] + [int(h) for h in hidden_dims] + [int(action_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal_xy], dim=-1)
        return self.net(x)


def build_gcbc_her_samples(
    observations: np.ndarray,
    actions: np.ndarray,
    timeouts: np.ndarray,
    *,
    geom_p: float,
    min_k: int,
    max_k: int,
    min_distance: float,
    her_k_per_transition: int,
    future_sample_attempts: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    t = np.asarray(timeouts, dtype=np.bool_)
    if len(obs) == 0:
        raise ValueError("Cannot build HER samples from empty observations")
    if len(obs) != len(act) or len(obs) != len(t):
        raise ValueError("observations/actions/timeouts length mismatch")
    if her_k_per_transition <= 0:
        raise ValueError("her_k_per_transition must be > 0")
    if future_sample_attempts <= 0:
        raise ValueError("future_sample_attempts must be > 0")

    rng = np.random.default_rng(seed)
    spans = dprobe.episode_spans_from_timeouts(t)

    xs_obs: List[np.ndarray] = []
    xs_goal: List[np.ndarray] = []
    ys_act: List[np.ndarray] = []

    transitions_total = 0
    transitions_with_any_her = 0
    short_span_skips = 0
    relabels_accepted = 0
    relabels_attempted = 0

    for s0, s1 in spans:
        for i in range(s0, s1):
            transitions_total += 1
            max_local_k = min(int(max_k), int(s1 - 1 - i))
            if max_local_k < 1:
                short_span_skips += 1
                continue

            added = 0
            tries_left = int(max(1, her_k_per_transition * future_sample_attempts))
            while added < her_k_per_transition and tries_left > 0:
                tries_left -= 1
                relabels_attempted += 1

                if max_local_k >= min_k:
                    k = dprobe.sample_truncated_geometric_k(
                        rng=rng,
                        p=float(geom_p),
                        min_k=int(min_k),
                        max_k=int(max_local_k),
                    )
                    g_idx = int(i + k)
                else:
                    # Fallback to any future state inside this episode.
                    g_idx = int(i + int(rng.integers(1, max_local_k + 1)))

                goal_xy = obs[g_idx, :2].astype(np.float32)
                dist = float(np.linalg.norm(goal_xy - obs[i, :2]))
                if dist < float(min_distance):
                    continue

                xs_obs.append(obs[i].copy())
                xs_goal.append(goal_xy.copy())
                ys_act.append(act[i].copy())
                added += 1
                relabels_accepted += 1

            # Deterministic fallback so every eligible transition contributes.
            if added == 0:
                g_idx = int(i + max_local_k)
                goal_xy = obs[g_idx, :2].astype(np.float32)
                dist = float(np.linalg.norm(goal_xy - obs[i, :2]))
                if dist < float(min_distance) and (i + 1) < s1:
                    goal_xy = obs[i + 1, :2].astype(np.float32)
                xs_obs.append(obs[i].copy())
                xs_goal.append(goal_xy.copy())
                ys_act.append(act[i].copy())
                added += 1
                relabels_accepted += 1

            if added > 0:
                transitions_with_any_her += 1

    if len(xs_obs) == 0:
        # Absolute fallback for pathological settings: one-step future goals.
        for s0, s1 in spans:
            for i in range(s0, max(s0, s1 - 1)):
                g_idx = min(i + 1, s1 - 1)
                xs_obs.append(obs[i].copy())
                xs_goal.append(obs[g_idx, :2].astype(np.float32).copy())
                ys_act.append(act[i].copy())
        if len(xs_obs) == 0:
            raise RuntimeError(
                "No GCBC+HER samples were produced. Relax distance/geom constraints or increase episode length."
            )

    her_stats = {
        "her_samples": int(len(xs_obs)),
        "her_relabels_attempted": int(relabels_attempted),
        "her_relabels_accepted": int(relabels_accepted),
        "her_accept_rate": float(relabels_accepted / max(1, relabels_attempted)),
        "her_transitions_total": int(transitions_total),
        "her_transitions_with_samples": int(transitions_with_any_her),
        "her_short_span_skips": int(short_span_skips),
        "her_samples_per_transition_mean": float(len(xs_obs) / max(1, transitions_total)),
    }
    return (
        np.asarray(xs_obs, dtype=np.float32),
        np.asarray(xs_goal, dtype=np.float32),
        np.asarray(ys_act, dtype=np.float32),
        her_stats,
    )


def build_gcbc_splits(
    raw_dataset: Dict[str, np.ndarray],
    cfg: Config,
    *,
    split_seed: int,
) -> Tuple[
    GCBCHERDataset,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    np.ndarray,
    np.ndarray,
    Dict[str, float],
]:
    xs_obs, xs_goal, ys_act, her_stats = build_gcbc_her_samples(
        observations=raw_dataset["observations"],
        actions=raw_dataset["actions"],
        timeouts=raw_dataset["timeouts"],
        geom_p=float(cfg.online_goal_geom_p),
        min_k=int(cfg.online_goal_geom_min_k),
        max_k=int(cfg.online_goal_geom_max_k),
        min_distance=float(cfg.online_goal_min_distance),
        her_k_per_transition=int(cfg.gcbc_her_k_per_transition),
        future_sample_attempts=int(cfg.gcbc_future_sample_attempts),
        seed=int(split_seed),
    )
    dataset = GCBCHERDataset(xs_obs, xs_goal, ys_act)

    rng = np.random.default_rng(split_seed)
    all_idx = np.arange(len(dataset))
    rng.shuffle(all_idx)
    if len(all_idx) < 2:
        train_idx = all_idx.copy()
        val_idx = all_idx.copy()
    else:
        n_val = min(len(all_idx) - 1, max(1, int(len(all_idx) * float(cfg.val_frac))))
        val_idx = all_idx[:n_val]
        train_idx = all_idx[n_val:]

    train_subset = torch.utils.data.Subset(dataset, indices=train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, indices=val_idx.tolist())
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, train_loader, val_loader, train_idx, val_idx, her_stats


@torch.no_grad()
def compute_val_loss(
    policy: GCBCPolicy,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_batches: int,
) -> float:
    policy.eval()
    losses: List[float] = []
    mse = nn.MSELoss()
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        obs, goal_xy, act = [x.to(device=device, dtype=torch.float32) for x in batch]
        pred = policy(obs, goal_xy)
        losses.append(float(mse(pred, act).item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def policy_action(
    policy: GCBCPolicy,
    obs: np.ndarray,
    goal_xy: np.ndarray,
    *,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device)
    goal_t = torch.as_tensor(np.asarray(goal_xy, dtype=np.float32)[None, :], device=device)
    act = policy(obs_t, goal_t)[0].detach().cpu().numpy().astype(np.float32)
    return np.clip(act, action_low, action_high).astype(np.float32)


@torch.no_grad()
def rollout_policy_to_goal(
    policy: GCBCPolicy,
    rollout_env: gym.Env,
    *,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rollout_horizon: int,
    rollout_mode: str,
    decision_every_n_steps: int,
    device: torch.device,
    maze_arr: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    obs = dprobe.reset_rollout_start(rollout_env, start_xy=start_xy)
    min_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
    final_goal_dist = min_goal_dist
    rollout_wall_hits = int(dprobe.count_wall_hits_qpos_frame(maze_arr, obs[:2]))

    act_low = np.asarray(rollout_env.action_space.low, dtype=np.float32)
    act_high = np.asarray(rollout_env.action_space.high, dtype=np.float32)
    stride = max(1, int(decision_every_n_steps))
    if rollout_mode == "open_loop":
        stride = max(1, int(rollout_horizon))

    rollout_xy: List[np.ndarray] = [obs[:2].copy()]
    rollout_actions: List[np.ndarray] = []
    obs_full: List[np.ndarray] = [obs.copy()]
    cached_action: np.ndarray | None = None
    decision_count = 0

    for t in range(int(rollout_horizon)):
        if cached_action is None or (t % stride == 0):
            cached_action = policy_action(
                policy=policy,
                obs=obs,
                goal_xy=goal_xy,
                device=device,
                action_low=act_low,
                action_high=act_high,
            )
            decision_count += 1
        action = cached_action.copy()

        obs, _, _, _ = dprobe.safe_step(rollout_env, action)
        dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = min(min_goal_dist, dist)
        final_goal_dist = dist
        rollout_wall_hits += int(dprobe.count_wall_hits_qpos_frame(maze_arr, obs[:2]))

        rollout_actions.append(action.copy())
        rollout_xy.append(obs[:2].copy())
        obs_full.append(obs.copy())

    return (
        np.asarray(rollout_xy, dtype=np.float32),
        np.asarray(rollout_actions, dtype=np.float32),
        np.asarray(obs_full, dtype=np.float32),
        float(min_goal_dist),
        float(final_goal_dist),
        int(rollout_wall_hits),
        int(decision_count),
    )


@torch.no_grad()
def evaluate_goal_progress_gcbc(
    policy: GCBCPolicy,
    *,
    env_name: str,
    query_pairs: List[Tuple[np.ndarray, np.ndarray]],
    rollout_horizon: int,
    success_prefix_horizons: Sequence[int],
    device: torch.device,
    n_samples: int,
    goal_success_threshold: float,
    rollout_mode: str,
    rollout_replan_every_n_steps: int,
    maze_arr: np.ndarray | None,
) -> Dict[str, float]:
    imagined_successes: List[float] = []
    imagined_goal_errors: List[float] = []
    imagined_pregoal_successes: List[float] = []
    imagined_pregoal_errors: List[float] = []
    imagined_line_dev: List[float] = []
    imagined_path_ratio: List[float] = []
    start_jump_ratios: List[float] = []
    end_jump_ratios: List[float] = []
    state_velocity_l2_means: List[float] = []
    state_velocity_rel_means: List[float] = []
    velocity_action_l2_means: List[float] = []
    velocity_action_rel_means: List[float] = []

    rollout_successes: List[float] = []
    rollout_final_goal_errors: List[float] = []
    rollout_min_goal_distances: List[float] = []
    rollout_successes_by_prefix: Dict[int, List[float]] = {int(h): [] for h in success_prefix_horizons}
    rollout_min_goal_dist_by_prefix: Dict[int, List[float]] = {int(h): [] for h in success_prefix_horizons}
    rollout_final_goal_dist_by_prefix: Dict[int, List[float]] = {int(h): [] for h in success_prefix_horizons}
    query_success_any_by_prefix: Dict[int, np.ndarray] = {
        int(h): np.zeros(len(query_pairs), dtype=np.bool_) for h in success_prefix_horizons
    }
    query_goal_cells: List[Tuple[int, int] | None] = [
        dprobe.qpos_xy_to_maze_cell(goal_xy, maze_arr=maze_arr) for _, goal_xy in query_pairs
    ]
    imagined_wall_hits: List[int] = []
    rollout_wall_hits: List[int] = []
    decision_counts: List[int] = []

    rollout_env = gym.make(env_name)
    dt = float(getattr(rollout_env.unwrapped, "dt", 1.0))
    for qid, (start_xy, goal_xy) in enumerate(query_pairs):
        for _ in range(int(n_samples)):
            rollout_xy, rollout_actions, obs_full, min_goal_dist, final_goal_dist, roll_wall, decisions = rollout_policy_to_goal(
                policy=policy,
                rollout_env=rollout_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                rollout_horizon=rollout_horizon,
                rollout_mode=rollout_mode,
                decision_every_n_steps=rollout_replan_every_n_steps,
                device=device,
                maze_arr=maze_arr,
            )

            imagined_obs = obs_full[:-1]  # length aligns with actions.
            imagined_xy = imagined_obs[:, :2]
            s = dprobe.straightness_metrics(xy=imagined_xy, start_xy=start_xy, goal_xy=goal_xy)
            b = dprobe.boundary_jump_ratios(imagined_xy)
            c = dprobe.transition_compatibility_metrics(
                observations=imagined_obs,
                actions=rollout_actions,
                goal_xy=goal_xy,
                dt=dt,
                goal_success_threshold=goal_success_threshold,
            )

            imagined_goal_error = float(s["final_goal_error"])
            imagined_successes.append(float(imagined_goal_error <= goal_success_threshold))
            imagined_goal_errors.append(imagined_goal_error)
            imagined_pregoal_successes.append(float(c["pregoal_success"]))
            imagined_pregoal_errors.append(float(c["pregoal_error"]))
            imagined_line_dev.append(float(s["mean_line_deviation"]))
            imagined_path_ratio.append(float(s["path_over_direct"]))
            start_jump_ratios.append(float(b["start_jump_ratio"]))
            end_jump_ratios.append(float(b["end_jump_ratio"]))
            state_velocity_l2_means.append(float(c["state_velocity_l2_mean"]))
            state_velocity_rel_means.append(float(c["state_velocity_rel_mean"]))
            velocity_action_l2_means.append(float(c["velocity_action_l2_mean"]))
            velocity_action_rel_means.append(float(c["velocity_action_rel_mean"]))
            imagined_wall_hits.append(int(dprobe.count_wall_hits_qpos_frame(maze_arr, imagined_xy)))

            prefix_stats = dprobe.rollout_prefix_distance_stats(
                rollout_xy=rollout_xy,
                goal_xy=goal_xy,
                prefix_horizons=success_prefix_horizons,
            )
            for h in success_prefix_horizons:
                h_int = int(h)
                prefix_min_goal_dist, prefix_final_goal_dist = prefix_stats[h_int]
                hit = bool(prefix_min_goal_dist <= goal_success_threshold)
                rollout_successes_by_prefix[h_int].append(float(hit))
                rollout_min_goal_dist_by_prefix[h_int].append(float(prefix_min_goal_dist))
                rollout_final_goal_dist_by_prefix[h_int].append(float(prefix_final_goal_dist))
                if hit:
                    query_success_any_by_prefix[h_int][qid] = True

            rollout_successes.append(float(min_goal_dist <= goal_success_threshold))
            rollout_min_goal_distances.append(float(min_goal_dist))
            rollout_final_goal_errors.append(float(final_goal_dist))
            rollout_wall_hits.append(int(roll_wall))
            decision_counts.append(int(decisions))
    rollout_env.close()

    eval_num_queries = int(len(query_pairs))
    eval_samples_per_query = int(n_samples)
    eval_num_trajectories = int(eval_num_queries * eval_samples_per_query)
    rollout_success_count = int(np.sum(rollout_successes)) if rollout_successes else 0
    goal_cell_total = int(len({cell for cell in query_goal_cells if cell is not None}))

    metrics: Dict[str, float] = {
        "eval_num_queries": eval_num_queries,
        "eval_samples_per_query": eval_samples_per_query,
        "eval_num_trajectories": eval_num_trajectories,
        "eval_unique_goal_cells": goal_cell_total,
        "eval_rollout_mode": rollout_mode,
        "eval_rollout_replan_every_n_steps": int(rollout_replan_every_n_steps),
        "eval_rollout_horizon": int(rollout_horizon),
        "eval_success_prefix_horizons": ",".join(str(int(h)) for h in success_prefix_horizons),
        "imagined_goal_success_rate": float(np.mean(imagined_successes)) if imagined_successes else float("nan"),
        "imagined_goal_error_mean": float(np.mean(imagined_goal_errors)) if imagined_goal_errors else float("nan"),
        "imagined_pregoal_success_rate": float(np.mean(imagined_pregoal_successes)) if imagined_pregoal_successes else float("nan"),
        "imagined_pregoal_error_mean": float(np.mean(imagined_pregoal_errors)) if imagined_pregoal_errors else float("nan"),
        "imagined_line_deviation_mean": float(np.mean(imagined_line_dev)) if imagined_line_dev else float("nan"),
        "imagined_path_over_direct_mean": float(np.mean(imagined_path_ratio)) if imagined_path_ratio else float("nan"),
        "boundary_start_jump_ratio_mean": float(np.mean(start_jump_ratios)) if start_jump_ratios else float("nan"),
        "boundary_end_jump_ratio_mean": float(np.mean(end_jump_ratios)) if end_jump_ratios else float("nan"),
        "state_velocity_l2_mean": float(np.mean(state_velocity_l2_means)) if state_velocity_l2_means else float("nan"),
        "state_velocity_rel_mean": float(np.mean(state_velocity_rel_means)) if state_velocity_rel_means else float("nan"),
        "velocity_action_l2_mean": float(np.mean(velocity_action_l2_means)) if velocity_action_l2_means else float("nan"),
        "velocity_action_rel_mean": float(np.mean(velocity_action_rel_means)) if velocity_action_rel_means else float("nan"),
        "rollout_goal_success_rate": float(np.mean(rollout_successes)) if rollout_successes else float("nan"),
        "rollout_success_count": rollout_success_count,
        "rollout_final_goal_error_mean": float(np.mean(rollout_final_goal_errors)) if rollout_final_goal_errors else float("nan"),
        "rollout_min_goal_distance_mean": float(np.mean(rollout_min_goal_distances)) if rollout_min_goal_distances else float("nan"),
        "imagined_in_wall_points_mean": float(np.mean(imagined_wall_hits)) if imagined_wall_hits else float("nan"),
        "rollout_in_wall_points_mean": float(np.mean(rollout_wall_hits)) if rollout_wall_hits else float("nan"),
        "policy_decisions_per_rollout_mean": float(np.mean(decision_counts)) if decision_counts else float("nan"),
    }

    for h in success_prefix_horizons:
        h_int = int(h)
        succ_list = rollout_successes_by_prefix[h_int]
        min_list = rollout_min_goal_dist_by_prefix[h_int]
        fin_list = rollout_final_goal_dist_by_prefix[h_int]
        query_hits = query_success_any_by_prefix[h_int]
        metrics[f"rollout_goal_success_rate_h{h_int}"] = float(np.mean(succ_list)) if succ_list else float("nan")
        metrics[f"rollout_success_count_h{h_int}"] = int(np.sum(succ_list)) if succ_list else 0
        metrics[f"rollout_min_goal_distance_mean_h{h_int}"] = float(np.mean(min_list)) if min_list else float("nan")
        metrics[f"rollout_final_goal_error_mean_h{h_int}"] = float(np.mean(fin_list)) if fin_list else float("nan")
        metrics[f"rollout_goal_query_coverage_rate_h{h_int}"] = (
            float(np.mean(query_hits.astype(np.float32))) if len(query_hits) else float("nan")
        )
        metrics[f"rollout_goal_query_coverage_count_h{h_int}"] = int(np.sum(query_hits)) if len(query_hits) else 0
        if goal_cell_total > 0:
            reached_goal_cells = {
                query_goal_cells[qidx]
                for qidx, hit in enumerate(query_hits)
                if bool(hit) and query_goal_cells[qidx] is not None
            }
            metrics[f"rollout_goal_cell_coverage_rate_h{h_int}"] = float(len(reached_goal_cells) / goal_cell_total)
            metrics[f"rollout_goal_cell_coverage_count_h{h_int}"] = int(len(reached_goal_cells))
        else:
            metrics[f"rollout_goal_cell_coverage_rate_h{h_int}"] = float("nan")
            metrics[f"rollout_goal_cell_coverage_count_h{h_int}"] = 0
    metrics["rollout_goal_cell_total"] = int(goal_cell_total)
    return metrics


@torch.no_grad()
def collect_policy_dataset(
    policy: GCBCPolicy,
    *,
    env_name: str,
    replay_observations: np.ndarray,
    replay_timeouts: np.ndarray,
    device: torch.device,
    n_episodes: int,
    episode_len: int,
    transition_budget: int,
    decision_every_n_steps: int,
    goal_geom_p: float,
    goal_geom_min_k: int,
    goal_geom_max_k: int,
    goal_min_distance: float,
    seed: int,
    maze_arr: np.ndarray | None,
    planning_success_thresholds: Sequence[float],
    planning_success_rel_reduction: float,
    early_terminate_on_success: bool,
    early_terminate_threshold: float,
    min_accepted_episode_len: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    stride = max(1, int(decision_every_n_steps))
    act_low = np.asarray(env.action_space.low, dtype=np.float32)
    act_high = np.asarray(env.action_space.high, dtype=np.float32)

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminals: List[bool] = []
    timeouts: List[bool] = []

    episode_lengths: List[int] = []
    goal_distances: List[float] = []
    goal_ks: List[int] = []
    decisions_per_episode: List[int] = []
    rollout_min_goal_dist: List[float] = []
    rollout_final_goal_dist: List[float] = []
    rollout_wall_hits: List[int] = []
    initial_goal_distances: List[float] = []
    final_distance_reduction_ratios: List[float] = []
    success_rel_reduction_flags: List[float] = []
    success_threshold_flags: Dict[float, List[float]] = {
        float(thr): [] for thr in planning_success_thresholds
    }

    accepted_episodes = 0
    accepted_transitions = 0
    attempted_episodes = 0
    attempted_transitions = 0
    rejected_short_episodes = 0
    min_len = max(1, int(min_accepted_episode_len))

    if transition_budget < 0:
        raise ValueError("transition_budget must be >= 0")
    if early_terminate_threshold <= 0.0:
        raise ValueError("early_terminate_threshold must be > 0")
    if transition_budget > 0:
        est_min_eps = int(math.ceil(float(transition_budget) / float(min_len)))
        max_attempts = max(1, est_min_eps * 10)
    else:
        max_attempts = max(1, int(n_episodes) * 10)

    while True:
        if transition_budget > 0:
            if accepted_transitions >= transition_budget:
                break
        else:
            if accepted_episodes >= n_episodes:
                break
        attempted_episodes += 1
        if attempted_episodes > max_attempts:
            raise RuntimeError(
                "Online collection exceeded max attempts while trying to gather accepted data "
                f"(accepted_eps={accepted_episodes}, accepted_transitions={accepted_transitions}, "
                f"attempted_eps={attempted_episodes}, rejected_short={rejected_short_episodes})."
            )

        start_xy, goal_xy, goal_k, sampled_goal_dist = dprobe.sample_geometric_start_goal_pair(
            observations=replay_observations,
            timeouts=replay_timeouts,
            rng=rng,
            geom_p=goal_geom_p,
            min_k=goal_geom_min_k,
            max_k=goal_geom_max_k,
            min_distance=goal_min_distance,
        )
        obs = dprobe.reset_rollout_start(env, start_xy=start_xy)
        initial_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = initial_goal_dist
        final_goal_dist = initial_goal_dist
        decision_count = 0
        cached_action: np.ndarray | None = None
        ep_len = 0

        ep_observations: List[np.ndarray] = []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []
        ep_terminals: List[bool] = []
        ep_timeouts: List[bool] = []
        ep_rollout_wall_hits: List[int] = []
        ep_dists_after_step: List[float] = []

        for t in range(int(episode_len)):
            should_decide = cached_action is None or (t % stride == 0)
            if should_decide:
                cached_action = policy_action(
                    policy=policy,
                    obs=obs,
                    goal_xy=goal_xy,
                    device=device,
                    action_low=act_low,
                    action_high=act_high,
                )
                decision_count += 1
            action = cached_action.copy()

            next_obs, reward, done, _ = dprobe.safe_step(env, action)
            attempted_transitions += 1
            dist = float(np.linalg.norm(next_obs[:2] - goal_xy))
            min_goal_dist = min(min_goal_dist, dist)
            final_goal_dist = dist
            wall_hit = int(dprobe.count_wall_hits_qpos_frame(maze_arr, next_obs[:2]))

            ep_observations.append(obs.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(float(reward))
            ep_terminals.append(bool(done))
            ep_rollout_wall_hits.append(wall_hit)
            ep_dists_after_step.append(dist)

            hit_goal = bool(early_terminate_on_success and (dist <= float(early_terminate_threshold)))
            is_timeout = bool((t == episode_len - 1) or done or hit_goal)
            ep_timeouts.append(is_timeout)

            obs = next_obs
            ep_len += 1
            if done or hit_goal:
                break

        if ep_len > 0:
            ep_timeouts[-1] = True

        if ep_len < min_len:
            rejected_short_episodes += 1
            continue

        if transition_budget > 0:
            remaining = int(transition_budget - accepted_transitions)
            if remaining <= 0:
                break
            if ep_len > remaining:
                if remaining < min_len:
                    break
                ep_observations = ep_observations[:remaining]
                ep_actions = ep_actions[:remaining]
                ep_rewards = ep_rewards[:remaining]
                ep_terminals = ep_terminals[:remaining]
                ep_timeouts = ep_timeouts[:remaining]
                ep_rollout_wall_hits = ep_rollout_wall_hits[:remaining]
                ep_dists_after_step = ep_dists_after_step[:remaining]
                ep_timeouts[-1] = True
                ep_len = remaining
                min_goal_dist = (
                    float(min(initial_goal_dist, float(np.min(ep_dists_after_step))))
                    if ep_dists_after_step
                    else float(initial_goal_dist)
                )
                final_goal_dist = float(ep_dists_after_step[-1]) if ep_dists_after_step else float("inf")

        observations.extend(ep_observations)
        actions.extend(ep_actions)
        rewards.extend(ep_rewards)
        terminals.extend(ep_terminals)
        timeouts.extend(ep_timeouts)

        accepted_episodes += 1
        accepted_transitions += ep_len
        episode_lengths.append(ep_len)
        goal_distances.append(sampled_goal_dist)
        goal_ks.append(goal_k)
        decisions_per_episode.append(int(decision_count))
        rollout_min_goal_dist.append(min_goal_dist)
        rollout_final_goal_dist.append(final_goal_dist)
        rollout_wall_hits.append(int(sum(ep_rollout_wall_hits)))
        initial_goal_distances.append(initial_goal_dist)

        if initial_goal_dist > 1e-8:
            reduction_ratio = float((initial_goal_dist - final_goal_dist) / initial_goal_dist)
        else:
            reduction_ratio = float(final_goal_dist <= 1e-6)
        final_distance_reduction_ratios.append(reduction_ratio)
        success_rel_reduction_flags.append(float(reduction_ratio >= planning_success_rel_reduction))
        for thr in planning_success_thresholds:
            success_threshold_flags[float(thr)].append(float(final_goal_dist <= float(thr)))

    env.close()

    new_dataset = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "timeouts": np.asarray(timeouts, dtype=np.bool_),
    }
    stats: Dict[str, float] = {
        "episodes": int(accepted_episodes),
        "transitions": int(len(new_dataset["observations"])),
        "episodes_attempted": int(attempted_episodes),
        "transitions_attempted": int(attempted_transitions),
        "episodes_rejected_short": int(rejected_short_episodes),
        "min_accepted_episode_len": int(min_len),
        "transition_budget": int(transition_budget),
        "episode_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episode_len_min": int(np.min(episode_lengths)) if episode_lengths else 0,
        "episode_len_max": int(np.max(episode_lengths)) if episode_lengths else 0,
        "sampled_goal_distance_mean": float(np.mean(goal_distances)) if goal_distances else float("nan"),
        "sampled_goal_k_mean": float(np.mean(goal_ks)) if goal_ks else float("nan"),
        "replans_per_episode_mean": float(np.mean(decisions_per_episode)) if decisions_per_episode else 0.0,
        "selected_plan_wall_hits_mean": float("nan"),
        "rollout_wall_hits_mean": float(np.mean(rollout_wall_hits)) if rollout_wall_hits else float("nan"),
        "initial_goal_distance_mean": float(np.mean(initial_goal_distances)) if initial_goal_distances else float("nan"),
        "rollout_min_goal_distance_mean": float(np.mean(rollout_min_goal_dist)) if rollout_min_goal_dist else float("nan"),
        "rollout_final_goal_distance_mean": float(np.mean(rollout_final_goal_dist)) if rollout_final_goal_dist else float("nan"),
        "planning_success_thresholds": ",".join([f"{float(thr):.4f}" for thr in planning_success_thresholds]),
        "planning_success_rel_reduction_target": float(planning_success_rel_reduction),
        "planning_final_distance_reduction_ratio_mean": (
            float(np.mean(final_distance_reduction_ratios)) if final_distance_reduction_ratios else float("nan")
        ),
        "planning_success_rate_final_rel_reduction": (
            float(np.mean(success_rel_reduction_flags)) if success_rel_reduction_flags else float("nan")
        ),
        f"planning_success_rate_final_rel{dprobe.threshold_tag(planning_success_rel_reduction)}": (
            float(np.mean(success_rel_reduction_flags)) if success_rel_reduction_flags else float("nan")
        ),
    }
    for thr in planning_success_thresholds:
        tag = dprobe.threshold_tag(float(thr))
        flags = success_threshold_flags[float(thr)]
        stats[f"planning_success_rate_final_t{tag}"] = float(np.mean(flags)) if flags else float("nan")
    return new_dataset, stats


def main() -> None:
    cfg = parse_args()
    dprobe.set_seed(cfg.seed)
    if cfg.eval_rollout_replan_every_n_steps <= 0:
        raise ValueError("--eval_rollout_replan_every_n_steps must be > 0")
    if cfg.online_replan_every_n_steps <= 0:
        raise ValueError("--online_replan_every_n_steps must be > 0")
    if cfg.online_goal_geom_min_k <= 0:
        raise ValueError("--online_goal_geom_min_k must be > 0")
    if cfg.online_goal_geom_max_k < cfg.online_goal_geom_min_k:
        raise ValueError("--online_goal_geom_max_k must be >= --online_goal_geom_min_k")
    if not (0.0 < cfg.online_goal_geom_p <= 1.0):
        raise ValueError("--online_goal_geom_p must be in (0, 1]")
    if cfg.eval_rollout_horizon <= 0:
        raise ValueError("--eval_rollout_horizon must be > 0")
    if cfg.online_collect_transition_budget_per_round < 0:
        raise ValueError("--online_collect_transition_budget_per_round must be >= 0")
    if cfg.online_early_terminate_threshold <= 0.0:
        raise ValueError("--online_early_terminate_threshold must be > 0")
    if cfg.online_min_accepted_episode_len < 0:
        raise ValueError("--online_min_accepted_episode_len must be >= 0")
    if cfg.gcbc_her_k_per_transition <= 0:
        raise ValueError("--gcbc_her_k_per_transition must be > 0")
    if cfg.gcbc_future_sample_attempts <= 0:
        raise ValueError("--gcbc_future_sample_attempts must be > 0")
    if not (0.0 <= cfg.online_planning_success_rel_reduction <= 1.0):
        raise ValueError("--online_planning_success_rel_reduction must be in [0, 1]")
    if cfg.eval_rollout_mode not in ("open_loop", "receding_horizon"):
        raise ValueError("--eval_rollout_mode must be open_loop or receding_horizon")

    online_min_accepted_episode_len = (
        int(cfg.online_min_accepted_episode_len)
        if int(cfg.online_min_accepted_episode_len) > 0
        else int(cfg.horizon)
    )
    if online_min_accepted_episode_len > int(cfg.online_collect_episode_len):
        raise ValueError(
            "--online_min_accepted_episode_len must be <= --online_collect_episode_len "
            f"(min_accepted={online_min_accepted_episode_len}, collect_episode_len={cfg.online_collect_episode_len})"
        )
    if (
        cfg.online_collect_transition_budget_per_round > 0
        and cfg.online_collect_transition_budget_per_round < online_min_accepted_episode_len
    ):
        raise ValueError(
            "--online_collect_transition_budget_per_round must be >= --online_min_accepted_episode_len "
            f"(budget={cfg.online_collect_transition_budget_per_round}, min_accepted={online_min_accepted_episode_len})"
        )

    eval_success_prefix_horizons = tuple(
        int(h) for h in dprobe.parse_int_list(cfg.eval_success_prefix_horizons, "--eval_success_prefix_horizons")
    )
    if any(h <= 0 for h in eval_success_prefix_horizons):
        raise ValueError("--eval_success_prefix_horizons must contain positive integers")
    if max(eval_success_prefix_horizons) > cfg.eval_rollout_horizon:
        raise ValueError(
            "--eval_success_prefix_horizons must not exceed --eval_rollout_horizon "
            f"(max prefix={max(eval_success_prefix_horizons)}, rollout_horizon={cfg.eval_rollout_horizon})"
        )
    online_planning_success_thresholds = tuple(
        float(thr)
        for thr in dprobe.parse_float_list(cfg.online_planning_success_thresholds, "--online_planning_success_thresholds")
    )
    if any(thr <= 0.0 for thr in online_planning_success_thresholds):
        raise ValueError("--online_planning_success_thresholds must be > 0")

    logdir = dprobe.make_logdir(cfg)
    with open(logdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] method=gcbc_her device={device}")
    print(f"[setup] logdir={logdir}")
    maze_arr = dprobe.load_maze_arr_from_env(cfg.env)
    if maze_arr is None:
        print("[setup] maze geometry unavailable.")
    else:
        print(
            f"[setup] maze geometry loaded: shape={maze_arr.shape} "
            f"wall_cells={int(np.sum(maze_arr == 10))}"
        )

    raw_dataset, action_low, action_high, collection_stats = dprobe.collect_random_dataset(
        env_name=cfg.env,
        n_episodes=cfg.n_episodes,
        episode_len=cfg.episode_len,
        action_scale=cfg.action_scale,
        seed=cfg.seed,
        corridor_aware_data=cfg.corridor_aware_data,
        corridor_max_resamples=cfg.corridor_max_resamples,
    )
    print(
        f"[data] transitions={len(raw_dataset['observations'])} "
        f"episodes={dprobe.count_episodes_from_timeouts(raw_dataset['timeouts'])}"
    )
    print(f"[data] action_low={action_low} action_high={action_high}")
    print(
        "[data] collection_stats "
        f"episode_len_mean={collection_stats['episode_len_mean']:.2f} "
        f"episode_len_min={collection_stats['episode_len_min']} "
        f"episode_len_max={collection_stats['episode_len_max']} "
        f"wall_rejects={collection_stats['wall_rejects']} "
        f"failed_steps={collection_stats['failed_steps']}"
    )

    (
        gcbc_dataset,
        train_loader,
        val_loader,
        train_idx,
        val_idx,
        her_stats,
    ) = build_gcbc_splits(raw_dataset=raw_dataset, cfg=cfg, split_seed=cfg.seed + 1337)
    initial_train_samples = int(len(train_idx))
    initial_val_samples = int(len(val_idx))
    print(
        f"[gcbc-her] samples total={len(gcbc_dataset)} "
        f"train={initial_train_samples} val={initial_val_samples} "
        f"her_samples_per_transition={her_stats['her_samples_per_transition_mean']:.3f}"
    )

    obs_dim = int(raw_dataset["observations"].shape[1])
    action_dim = int(raw_dataset["actions"].shape[1])
    hidden_dims = parse_hidden_dims(cfg.gcbc_hidden_dims)
    policy = GCBCPolicy(
        observation_dim=obs_dim,
        goal_dim=2,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    mse = nn.MSELoss()

    metrics_rows: List[Dict[str, float]] = []
    progress_rows: List[Dict[str, float]] = []
    online_collection_rows: List[Dict[str, float]] = []

    if cfg.query_mode == "fixed":
        query_bank: List[Tuple[np.ndarray, np.ndarray]] = dprobe.parse_queries(cfg.query)
        print(f"[eval-query] mode=fixed num_pairs={len(query_bank)}")
    else:
        bank_size = max(cfg.query_bank_size, cfg.num_eval_queries)
        query_bank = dprobe.build_diverse_query_bank(
            points_xy=raw_dataset["observations"][:, :2],
            bank_size=bank_size,
            n_angle_bins=cfg.query_angle_bins,
            min_pair_distance=cfg.query_min_distance,
            seed=cfg.seed + 991,
        )
        print(
            f"[eval-query] mode=diverse bank_size={len(query_bank)} "
            f"num_eval_queries={cfg.num_eval_queries} angle_bins={cfg.query_angle_bins} "
            f"min_distance={cfg.query_min_distance:.3f} "
            f"resample_each_eval={cfg.query_resample_each_eval}"
        )

    def query_pairs_for_step(step: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if cfg.query_mode == "fixed":
            return list(query_bank)
        if cfg.query_resample_each_eval:
            q_seed = cfg.seed + cfg.query_resample_seed_stride * max(step, 1)
        else:
            q_seed = cfg.seed + cfg.query_resample_seed_stride
        return dprobe.select_query_pairs(
            query_bank=query_bank,
            num_queries=cfg.num_eval_queries,
            seed=q_seed,
        )

    global_step = 0
    last_eval_query_pairs = query_pairs_for_step(step=0)

    def run_training_steps(
        num_steps: int,
        train_loader_cur: torch.utils.data.DataLoader,
        val_loader_cur: torch.utils.data.DataLoader,
        *,
        phase: str,
    ) -> None:
        nonlocal global_step, last_eval_query_pairs
        train_iter = dprobe.cycle(train_loader_cur)
        for _ in range(int(num_steps)):
            global_step += 1
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            obs_b, goal_b, act_b = next(train_iter)
            obs_b = obs_b.to(device=device, dtype=torch.float32)
            goal_b = goal_b.to(device=device, dtype=torch.float32)
            act_b = act_b.to(device=device, dtype=torch.float32)

            pred = policy(obs_b, goal_b)
            loss = mse(pred, act_b)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optimizer.step()

            val_loss = float("nan")
            if global_step == 1 or (cfg.val_every > 0 and global_step % cfg.val_every == 0):
                val_loss = compute_val_loss(
                    policy=policy,
                    val_loader=val_loader_cur,
                    device=device,
                    n_batches=cfg.val_batches,
                )

            row = {
                "step": int(global_step),
                "phase": phase,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
            }
            metrics_rows.append(row)

            if global_step == 1 or global_step % 100 == 0 or np.isfinite(val_loss):
                print(
                    f"[train] phase={phase} step={global_step:5d} "
                    f"train_loss={row['train_loss']:.5f} val_loss={row['val_loss']:.5f}"
                )

            if cfg.save_checkpoint_every > 0 and global_step % cfg.save_checkpoint_every == 0:
                checkpoint_step = {
                    "step": int(global_step),
                    "phase": phase,
                    "policy": policy.state_dict(),
                    "config": asdict(cfg),
                }
                torch.save(checkpoint_step, logdir / f"checkpoint_step{global_step}.pt")

            if cfg.eval_goal_every > 0 and global_step % cfg.eval_goal_every == 0:
                eval_query_pairs = query_pairs_for_step(step=global_step)
                last_eval_query_pairs = eval_query_pairs
                progress = evaluate_goal_progress_gcbc(
                    policy=policy,
                    env_name=cfg.env,
                    query_pairs=eval_query_pairs,
                    rollout_horizon=cfg.eval_rollout_horizon,
                    success_prefix_horizons=eval_success_prefix_horizons,
                    device=device,
                    n_samples=cfg.query_batch_size,
                    goal_success_threshold=cfg.goal_success_threshold,
                    rollout_mode=cfg.eval_rollout_mode,
                    rollout_replan_every_n_steps=cfg.eval_rollout_replan_every_n_steps,
                    maze_arr=maze_arr,
                )
                progress["step"] = int(global_step)
                progress["phase"] = phase
                progress["eval_query_mode"] = cfg.query_mode
                progress["eval_query_pairs"] = int(len(eval_query_pairs))
                progress_rows.append(progress)
                try:
                    pd.DataFrame(progress_rows).to_csv(logdir / "progress_metrics.csv", index=False)
                except Exception as e:
                    print(f"[warn] failed to flush progress_metrics.csv: {e}")

                short_h = int(eval_success_prefix_horizons[0])
                long_h = int(eval_success_prefix_horizons[-1])
                rollout_short = float(progress.get(f"rollout_goal_success_rate_h{short_h}", np.nan))
                rollout_long = float(progress.get(f"rollout_goal_success_rate_h{long_h}", np.nan))
                rollout_long_count = int(progress.get(f"rollout_success_count_h{long_h}", 0))
                cov_query_long = float(progress.get(f"rollout_goal_query_coverage_rate_h{long_h}", np.nan))
                cov_cell_long = float(progress.get(f"rollout_goal_cell_coverage_rate_h{long_h}", np.nan))
                print(
                    "[progress] "
                    f"phase={phase} step={global_step:5d} "
                    f"imagined_success={progress['imagined_goal_success_rate']:.3f} "
                    f"rollout_success@{short_h}={rollout_short:.3f} "
                    f"rollout_success@{long_h}={rollout_long:.3f} "
                    f"({rollout_long_count}/{int(progress['eval_num_trajectories'])}) "
                    f"goal_cov_query@{long_h}={cov_query_long:.3f} "
                    f"goal_cov_cell@{long_h}={cov_cell_long:.3f} "
                    f"imagined_goal_err={progress['imagined_goal_error_mean']:.3f} "
                    f"imagined_pregoal_err={progress['imagined_pregoal_error_mean']:.3f} "
                    f"rollout_goal_err={progress['rollout_final_goal_error_mean']:.3f} "
                    f"rollout_mode={progress['eval_rollout_mode']} "
                    f"query_pairs={int(progress['eval_query_pairs'])} "
                    f"imag_wall={progress['imagined_in_wall_points_mean']:.2f} "
                    f"roll_wall={progress['rollout_in_wall_points_mean']:.2f} "
                    f"state_vel_rel={progress['state_velocity_rel_mean']:.3f} "
                    f"vel_act_rel={progress['velocity_action_rel_mean']:.3f}"
                )

    run_training_steps(
        num_steps=cfg.train_steps,
        train_loader_cur=train_loader,
        val_loader_cur=val_loader,
        phase="offline_init",
    )

    if cfg.online_self_improve and cfg.online_rounds > 0:
        if cfg.online_collect_transition_budget_per_round > 0:
            collect_desc = f"collect_transition_budget_per_round={cfg.online_collect_transition_budget_per_round}"
        else:
            collect_desc = (
                f"collect_eps_per_round={cfg.online_collect_episodes_per_round} "
                f"collect_episode_len={cfg.online_collect_episode_len}"
            )
        print(
            "[online] enabled: "
            f"rounds={cfg.online_rounds} {collect_desc} "
            f"decision_every_n_steps={cfg.online_replan_every_n_steps} "
            f"train_steps_per_round={cfg.online_train_steps_per_round}"
        )
        for round_idx in range(1, cfg.online_rounds + 1):
            planner_dataset, planner_stats = collect_policy_dataset(
                policy=policy,
                env_name=cfg.env,
                replay_observations=raw_dataset["observations"],
                replay_timeouts=raw_dataset["timeouts"],
                device=device,
                n_episodes=cfg.online_collect_episodes_per_round,
                episode_len=cfg.online_collect_episode_len,
                transition_budget=cfg.online_collect_transition_budget_per_round,
                decision_every_n_steps=cfg.online_replan_every_n_steps,
                goal_geom_p=cfg.online_goal_geom_p,
                goal_geom_min_k=cfg.online_goal_geom_min_k,
                goal_geom_max_k=cfg.online_goal_geom_max_k,
                goal_min_distance=cfg.online_goal_min_distance,
                seed=cfg.seed + 10000 + round_idx,
                maze_arr=maze_arr,
                planning_success_thresholds=online_planning_success_thresholds,
                planning_success_rel_reduction=cfg.online_planning_success_rel_reduction,
                early_terminate_on_success=bool(cfg.online_early_terminate_on_success),
                early_terminate_threshold=float(cfg.online_early_terminate_threshold),
                min_accepted_episode_len=int(online_min_accepted_episode_len),
            )
            raw_dataset = dprobe.merge_replay_datasets(raw_dataset, planner_dataset)
            replay_transitions = int(len(raw_dataset["observations"]))
            replay_episodes = int(dprobe.count_episodes_from_timeouts(raw_dataset["timeouts"]))
            round_row = {
                "round": int(round_idx),
                "step_before_retrain": int(global_step),
                "replay_transitions": replay_transitions,
                "replay_episodes": replay_episodes,
                **planner_stats,
            }
            online_collection_rows.append(round_row)
            try:
                pd.DataFrame(online_collection_rows).to_csv(logdir / "online_collection.csv", index=False)
            except Exception as e:
                print(f"[warn] failed to flush online_collection.csv: {e}")
            threshold_summaries = " ".join(
                [
                    f"s@{thr:.2f}={planner_stats.get(f'planning_success_rate_final_t{dprobe.threshold_tag(thr)}', float('nan')):.3f}"
                    for thr in online_planning_success_thresholds
                ]
            )
            rel_tag = dprobe.threshold_tag(cfg.online_planning_success_rel_reduction)
            print(
                "[online-collect] "
                f"round={round_idx} "
                f"eps={planner_stats.get('episodes', 0)} "
                f"transitions={planner_stats['transitions']} "
                f"attempted_eps={planner_stats.get('episodes_attempted', 0)} "
                f"reject_short={planner_stats.get('episodes_rejected_short', 0)} "
                f"replay_transitions={replay_transitions} "
                f"sampled_goal_dist_mean={planner_stats['sampled_goal_distance_mean']:.3f} "
                f"sampled_goal_k_mean={planner_stats['sampled_goal_k_mean']:.2f} "
                f"decisions_per_ep={planner_stats['replans_per_episode_mean']:.2f} "
                f"roll_wall_hits={planner_stats['rollout_wall_hits_mean']:.2f} "
                f"rel{cfg.online_planning_success_rel_reduction:.2f}="
                f"{planner_stats.get(f'planning_success_rate_final_rel{rel_tag}', float('nan')):.3f} "
                f"{threshold_summaries}"
            )

            (
                gcbc_dataset,
                train_loader,
                val_loader,
                train_idx,
                val_idx,
                her_stats_round,
            ) = build_gcbc_splits(
                raw_dataset=raw_dataset,
                cfg=cfg,
                split_seed=cfg.seed + 1337 + round_idx,
            )
            print(
                "[online-replay] "
                f"round={round_idx} samples total={len(gcbc_dataset)} "
                f"train={len(train_idx)} val={len(val_idx)} "
                f"her_samples_per_transition={her_stats_round['her_samples_per_transition_mean']:.3f}"
            )

            run_training_steps(
                num_steps=cfg.online_train_steps_per_round,
                train_loader_cur=train_loader,
                val_loader_cur=val_loader,
                phase=f"online_round_{round_idx}",
            )
            her_stats = her_stats_round
    elif cfg.online_self_improve:
        print("[online] enabled but --online_rounds <= 0, skipping replay expansion rounds.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(logdir / "metrics.csv", index=False)
    dprobe.plot_losses(metrics_df, out_path=logdir / "train_val_loss.png")

    checkpoint = {
        "step": int(global_step),
        "policy": policy.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, logdir / "checkpoint_last.pt")

    progress_df = pd.DataFrame(progress_rows)
    progress_df.to_csv(logdir / "progress_metrics.csv", index=False)
    if len(online_collection_rows) > 0:
        pd.DataFrame(online_collection_rows).to_csv(logdir / "online_collection.csv", index=False)
    if len(progress_df) > 0:
        dprobe.plot_progress(progress_df, out_path=logdir / "goal_progress.png")

    final_query_pairs = (
        last_eval_query_pairs if len(last_eval_query_pairs) > 0 else query_pairs_for_step(step=global_step)
    )
    query_rows: List[Dict[str, float]] = []
    query_rollout_env = gym.make(cfg.env)
    for qid, (start_xy, goal_xy) in enumerate(final_query_pairs):
        for sid in range(int(cfg.query_batch_size)):
            rollout_xy, rollout_actions, obs_full, rollout_min_goal_dist, rollout_final_goal_dist, rollout_wall_hits, _ = rollout_policy_to_goal(
                policy=policy,
                rollout_env=query_rollout_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                rollout_horizon=cfg.eval_rollout_horizon,
                rollout_mode=cfg.eval_rollout_mode,
                decision_every_n_steps=cfg.eval_rollout_replan_every_n_steps,
                device=device,
                maze_arr=maze_arr,
            )
            imagined_obs = obs_full[:-1]
            xy = imagined_obs[:, :2]
            m = dprobe.straightness_metrics(xy=xy, start_xy=start_xy, goal_xy=goal_xy)
            b = dprobe.boundary_jump_ratios(xy)
            imagined_wall_hits = int(dprobe.count_wall_hits_qpos_frame(maze_arr, xy))
            query_rows.append(
                {
                    "query_id": qid,
                    "sample_id": sid,
                    "start_x": float(start_xy[0]),
                    "start_y": float(start_xy[1]),
                    "goal_x": float(goal_xy[0]),
                    "goal_y": float(goal_xy[1]),
                    "mean_line_deviation": float(m["mean_line_deviation"]),
                    "max_line_deviation": float(m["max_line_deviation"]),
                    "final_goal_error": float(m["final_goal_error"]),
                    "path_length": float(m["path_length"]),
                    "direct_distance": float(m["direct_distance"]),
                    "path_over_direct": float(m["path_over_direct"]),
                    "start_jump_ratio": float(b["start_jump_ratio"]),
                    "end_jump_ratio": float(b["end_jump_ratio"]),
                    "rollout_mode": cfg.eval_rollout_mode,
                    "rollout_min_goal_distance": float(rollout_min_goal_dist),
                    "rollout_final_goal_error": float(rollout_final_goal_dist),
                    "imagined_in_wall_points": int(imagined_wall_hits),
                    "rollout_in_wall_points": int(rollout_wall_hits),
                    "xy_json": json.dumps(xy.tolist()),
                    "action_json": json.dumps(rollout_actions.tolist()),
                    "rollout_xy_json": json.dumps(rollout_xy.tolist()),
                    "rollout_action_json": json.dumps(rollout_actions.tolist()),
                }
            )
    query_rollout_env.close()

    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(logdir / "query_metrics.csv", index=False)
    dprobe.plot_query_trajectories(
        query_rows=query_rows[: min(len(query_rows), 8)],
        out_path=logdir / "query_trajectories.png",
        maze_arr=maze_arr,
    )

    summary = {
        "method": "gcbc_her",
        "logdir": str(logdir),
        "train_steps_total": int(global_step),
        "online_self_improve": bool(cfg.online_self_improve),
        "online_rounds": int(cfg.online_rounds),
        "online_replan_every_n_steps": int(cfg.online_replan_every_n_steps),
        "dataset_transitions": int(len(raw_dataset["observations"])),
        "dataset_episodes": int(dprobe.count_episodes_from_timeouts(raw_dataset["timeouts"])),
        "dataset_collection_stats": collection_stats,
        "her_stats_last": her_stats,
        "initial_train_samples": int(initial_train_samples),
        "initial_val_samples": int(initial_val_samples),
        "final_train_samples": int(len(train_idx)),
        "final_val_samples": int(len(val_idx)),
        "final_train_loss": float(metrics_df["train_loss"].iloc[-1]),
        "final_val_loss": float(
            metrics_df[np.isfinite(metrics_df["val_loss"])]["val_loss"].iloc[-1]
            if np.isfinite(metrics_df["val_loss"]).any()
            else np.nan
        ),
        "eval_query_mode": cfg.query_mode,
        "eval_query_pairs_per_step": int(cfg.num_eval_queries if cfg.query_mode == "diverse" else len(query_bank)),
        "eval_query_bank_size": int(len(query_bank)),
        "eval_rollout_mode": cfg.eval_rollout_mode,
        "eval_rollout_replan_every_n_steps": int(cfg.eval_rollout_replan_every_n_steps),
        "eval_rollout_horizon": int(cfg.eval_rollout_horizon),
        "eval_success_prefix_horizons": [int(h) for h in eval_success_prefix_horizons],
        "online_planning_success_thresholds": [float(thr) for thr in online_planning_success_thresholds],
        "online_planning_success_rel_reduction": float(cfg.online_planning_success_rel_reduction),
        "query_path_over_direct_mean": float(query_df["path_over_direct"].mean()) if len(query_df) else np.nan,
        "query_mean_line_deviation_mean": float(query_df["mean_line_deviation"].mean()) if len(query_df) else np.nan,
        "query_rollout_min_goal_distance_mean": float(query_df["rollout_min_goal_distance"].mean()) if len(query_df) else np.nan,
        "query_rollout_final_goal_error_mean": float(query_df["rollout_final_goal_error"].mean()) if len(query_df) else np.nan,
        "query_imagined_in_wall_points_mean": float(query_df["imagined_in_wall_points"].mean()) if len(query_df) else np.nan,
        "query_rollout_in_wall_points_mean": float(query_df["rollout_in_wall_points"].mean()) if len(query_df) else np.nan,
        "progress_last": progress_rows[-1] if progress_rows else {},
        "online_collection_last": online_collection_rows[-1] if online_collection_rows else {},
    }
    with open(logdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n[done] artifacts:")
    for rel in [
        "config.json",
        "metrics.csv",
        "train_val_loss.png",
        "progress_metrics.csv",
        "online_collection.csv",
        "goal_progress.png",
        "query_metrics.csv",
        "query_trajectories.png",
        "checkpoint_last.pt",
        "summary.json",
    ]:
        if rel == "online_collection.csv" and len(online_collection_rows) == 0:
            continue
        print(f"  - {logdir / rel}")
    print("[done] summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
