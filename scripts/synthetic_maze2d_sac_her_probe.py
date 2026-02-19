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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # Offline synthetic replay collection (random actions).
    n_episodes: int = 400
    episode_len: int = 192
    action_scale: float = 1.0
    corridor_aware_data: bool = False
    corridor_max_resamples: int = 200

    # Goal sampling knobs (shared protocol).
    horizon: int = 64  # Used as default minimum accepted episode length when 0.
    max_path_length: int = 256
    online_goal_geom_p: float = 0.08
    online_goal_geom_min_k: int = 8
    online_goal_geom_max_k: int = 96
    online_goal_min_distance: float = 0.5

    # SAC + HER.
    learning_rate: float = 3e-4
    batch_size: int = 256  # HER samples per update.
    grad_clip: float = 0.0
    train_steps: int = 4000
    gamma: float = 0.99
    tau: float = 0.005
    policy_update_every: int = 1
    target_update_every: int = 1
    sac_hidden_dims: str = "256,256"
    sac_log_std_min: float = -20.0
    sac_log_std_max: float = 2.0
    sac_auto_alpha: bool = True
    sac_alpha: float = 0.2
    sac_target_entropy: float = math.nan  # If NaN, use -action_dim.

    her_k_per_transition: int = 4
    her_future_sample_attempts: int = 16

    reward_mode: str = "sparse"  # sparse|shaped
    shaped_reward_scale: float = 1.0
    shaped_success_bonus: float = 0.0
    sparse_success_value: float = 1.0

    # Evaluation protocol (shared).
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

    # Online self-improvement loop (collector is SAC policy + HER training).
    online_self_improve: bool = False
    online_rounds: int = 0
    online_train_steps_per_round: int = 2000
    online_collect_episodes_per_round: int = 32
    online_collect_episode_len: int = 256
    online_collect_transition_budget_per_round: int = 0
    online_replan_every_n_steps: int = 8
    online_planning_success_thresholds: str = "0.1,0.2"
    online_planning_success_rel_reduction: float = 0.9
    online_early_terminate_on_success: bool = True
    online_early_terminate_threshold: float = 0.2
    online_min_accepted_episode_len: int = 0


def _parse_int_list(raw: str, name: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{name} must contain at least one integer")
    return vals


def _parse_hidden_dims(raw: str) -> Tuple[int, ...]:
    return _parse_int_list(raw, "--sac_hidden_dims")


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=(
            "Synthetic Maze2D experiment: Goal-conditioned SAC + HER, aligned with "
            "the diffuser probe eval/online protocol."
        )
    )
    p.add_argument("--env", type=str, default=Config.env)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--logdir", type=str, default=Config.logdir)

    p.add_argument("--n_episodes", type=int, default=Config.n_episodes)
    p.add_argument("--episode_len", type=int, default=Config.episode_len)
    p.add_argument("--action_scale", type=float, default=Config.action_scale)
    p.add_argument("--corridor_aware_data", dest="corridor_aware_data", action="store_true")
    p.add_argument("--no_corridor_aware_data", dest="corridor_aware_data", action="store_false")
    p.set_defaults(corridor_aware_data=Config.corridor_aware_data)
    p.add_argument("--corridor_max_resamples", type=int, default=Config.corridor_max_resamples)

    p.add_argument("--horizon", type=int, default=Config.horizon)
    p.add_argument("--max_path_length", type=int, default=Config.max_path_length)
    p.add_argument("--online_goal_geom_p", type=float, default=Config.online_goal_geom_p)
    p.add_argument("--online_goal_geom_min_k", type=int, default=Config.online_goal_geom_min_k)
    p.add_argument("--online_goal_geom_max_k", type=int, default=Config.online_goal_geom_max_k)
    p.add_argument("--online_goal_min_distance", type=float, default=Config.online_goal_min_distance)

    p.add_argument("--learning_rate", type=float, default=Config.learning_rate)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--grad_clip", type=float, default=Config.grad_clip)
    p.add_argument("--train_steps", type=int, default=Config.train_steps)
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--tau", type=float, default=Config.tau)
    p.add_argument("--policy_update_every", type=int, default=Config.policy_update_every)
    p.add_argument("--target_update_every", type=int, default=Config.target_update_every)
    p.add_argument("--sac_hidden_dims", type=str, default=Config.sac_hidden_dims)
    p.add_argument("--sac_log_std_min", type=float, default=Config.sac_log_std_min)
    p.add_argument("--sac_log_std_max", type=float, default=Config.sac_log_std_max)
    p.add_argument("--sac_auto_alpha", dest="sac_auto_alpha", action="store_true")
    p.add_argument("--no_sac_auto_alpha", dest="sac_auto_alpha", action="store_false")
    p.set_defaults(sac_auto_alpha=Config.sac_auto_alpha)
    p.add_argument("--sac_alpha", type=float, default=Config.sac_alpha)
    p.add_argument("--sac_target_entropy", type=float, default=Config.sac_target_entropy)

    p.add_argument("--her_k_per_transition", type=int, default=Config.her_k_per_transition)
    p.add_argument("--her_future_sample_attempts", type=int, default=Config.her_future_sample_attempts)

    p.add_argument(
        "--reward_mode",
        type=str,
        choices=["sparse", "shaped"],
        default=Config.reward_mode,
        help="sparse: 0/1 success reward; shaped: -distance to goal (optionally +bonus on success).",
    )
    p.add_argument("--shaped_reward_scale", type=float, default=Config.shaped_reward_scale)
    p.add_argument("--shaped_success_bonus", type=float, default=Config.shaped_success_bonus)
    p.add_argument("--sparse_success_value", type=float, default=Config.sparse_success_value)

    p.add_argument("--query", type=str, default=Config.query)
    p.add_argument("--query_mode", type=str, choices=["fixed", "diverse"], default=Config.query_mode)
    p.add_argument("--num_eval_queries", type=int, default=Config.num_eval_queries)
    p.add_argument("--query_bank_size", type=int, default=Config.query_bank_size)
    p.add_argument("--query_angle_bins", type=int, default=Config.query_angle_bins)
    p.add_argument("--query_min_distance", type=float, default=Config.query_min_distance)
    p.add_argument("--query_resample_each_eval", dest="query_resample_each_eval", action="store_true")
    p.add_argument("--no_query_resample_each_eval", dest="query_resample_each_eval", action="store_false")
    p.set_defaults(query_resample_each_eval=Config.query_resample_each_eval)
    p.add_argument("--query_resample_seed_stride", type=int, default=Config.query_resample_seed_stride)
    p.add_argument("--query_batch_size", type=int, default=Config.query_batch_size)
    p.add_argument("--eval_goal_every", type=int, default=Config.eval_goal_every)
    p.add_argument("--goal_success_threshold", type=float, default=Config.goal_success_threshold)
    p.add_argument("--eval_rollout_mode", type=str, choices=["open_loop", "receding_horizon"], default=Config.eval_rollout_mode)
    p.add_argument("--eval_rollout_replan_every_n_steps", type=int, default=Config.eval_rollout_replan_every_n_steps)
    p.add_argument("--eval_rollout_horizon", type=int, default=Config.eval_rollout_horizon)
    p.add_argument("--eval_success_prefix_horizons", type=str, default=Config.eval_success_prefix_horizons)
    p.add_argument("--save_checkpoint_every", type=int, default=Config.save_checkpoint_every)

    p.add_argument("--online_self_improve", dest="online_self_improve", action="store_true")
    p.add_argument("--no_online_self_improve", dest="online_self_improve", action="store_false")
    p.set_defaults(online_self_improve=Config.online_self_improve)
    p.add_argument("--online_rounds", type=int, default=Config.online_rounds)
    p.add_argument("--online_train_steps_per_round", type=int, default=Config.online_train_steps_per_round)
    p.add_argument("--online_collect_episodes_per_round", type=int, default=Config.online_collect_episodes_per_round)
    p.add_argument("--online_collect_episode_len", type=int, default=Config.online_collect_episode_len)
    p.add_argument("--online_collect_transition_budget_per_round", type=int, default=Config.online_collect_transition_budget_per_round)
    p.add_argument("--online_replan_every_n_steps", type=int, default=Config.online_replan_every_n_steps)
    p.add_argument("--online_planning_success_thresholds", type=str, default=Config.online_planning_success_thresholds)
    p.add_argument("--online_planning_success_rel_reduction", type=float, default=Config.online_planning_success_rel_reduction)
    p.add_argument("--online_early_terminate_on_success", dest="online_early_terminate_on_success", action="store_true")
    p.add_argument("--no_online_early_terminate_on_success", dest="online_early_terminate_on_success", action="store_false")
    p.set_defaults(online_early_terminate_on_success=Config.online_early_terminate_on_success)
    p.add_argument("--online_early_terminate_threshold", type=float, default=Config.online_early_terminate_threshold)
    p.add_argument("--online_min_accepted_episode_len", type=int, default=Config.online_min_accepted_episode_len)

    return Config(**vars(p.parse_args()))


def _build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential:
    dims = [int(in_dim)] + [int(h) for h in hidden_dims] + [int(out_dim)]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        in_dim = int(obs_dim + goal_dim)
        self.net = _build_mlp(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=int(2 * action_dim))

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor, goal_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, goal_xy], dim=-1)
        out = self.net(x)
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor, goal_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs, goal_xy)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias

        # Log-prob with tanh + affine correction.
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        # Jacobian of a = scale * tanh(raw) + bias is scale*(1 - tanh^2(raw)).
        eps = 1e-6
        log_det = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + eps).sum(dim=-1, keepdim=True)
        log_prob = log_prob - log_det
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def act_deterministic(self, obs: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs, goal_xy)
        squashed = torch.tanh(mean)
        return squashed * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        in_dim = int(obs_dim + goal_dim + action_dim)
        self.net = _build_mlp(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=1)

    def forward(self, obs: torch.Tensor, goal_xy: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal_xy, action], dim=-1)
        return self.net(x)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    t = float(tau)
    with torch.no_grad():
        # Python 3.8 compatibility: zip(..., strict=True) not available.
        tgt_params = list(target.parameters())
        src_params = list(source.parameters())
        if len(tgt_params) != len(src_params):
            raise ValueError("target/source parameter length mismatch in soft update")
        for tp, sp in zip(tgt_params, src_params):
            tp.data.mul_(1.0 - t).add_(sp.data, alpha=t)


def _compute_episode_end_indices(timeouts: np.ndarray) -> np.ndarray:
    t = np.asarray(timeouts, dtype=np.bool_)
    end_idx = np.empty(len(t), dtype=np.int64)
    start = 0
    for i, timeout in enumerate(t):
        if bool(timeout):
            end = i + 1
            end_idx[start:end] = end
            start = end
    if start < len(t):
        end_idx[start:] = len(t)
    return end_idx


def _sample_goal_index_for_transition(
    rng: np.random.Generator,
    *,
    obs: np.ndarray,
    end_idx: np.ndarray,
    i: int,
    geom_p: float,
    min_k: int,
    max_k: int,
    min_distance: float,
    attempts: int,
) -> int:
    n = len(obs)
    i = int(i)
    if i < 0 or i >= n:
        raise IndexError("transition index out of bounds")
    ep_end = int(end_idx[i])
    max_local_k = min(int(max_k), int(ep_end - 1 - i))
    if max_local_k < 1:
        return i

    for _ in range(max(1, int(attempts))):
        if max_local_k >= int(min_k):
            k = dprobe.sample_truncated_geometric_k(
                rng=rng,
                p=float(geom_p),
                min_k=int(min_k),
                max_k=int(max_local_k),
            )
            g_idx = int(i + k)
        else:
            g_idx = int(i + int(rng.integers(1, max_local_k + 1)))
        goal_xy = obs[g_idx, :2].astype(np.float32)
        dist = float(np.linalg.norm(goal_xy - obs[i, :2]))
        if dist >= float(min_distance):
            return g_idx

    # Deterministic fallback so every transition yields a goal.
    g_idx = int(i + max_local_k)
    if g_idx <= i and (i + 1) < ep_end:
        g_idx = i + 1
    return int(min(g_idx, ep_end - 1))


def _build_her_sac_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    batch_size: int,
    goal_success_threshold: float,
    reward_mode: str,
    shaped_reward_scale: float,
    shaped_success_bonus: float,
    sparse_success_value: float,
    geom_p: float,
    min_k: int,
    max_k: int,
    min_distance: float,
    her_k_per_transition: int,
    future_sample_attempts: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    term = np.asarray(terminals, dtype=np.bool_)
    tout = np.asarray(timeouts, dtype=np.bool_)
    if len(obs) == 0:
        raise ValueError("empty replay")
    if len(obs) != len(act) or len(obs) != len(term) or len(obs) != len(tout):
        raise ValueError("replay arrays must have equal length")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if her_k_per_transition <= 0:
        raise ValueError("her_k_per_transition must be > 0")
    if future_sample_attempts <= 0:
        raise ValueError("future_sample_attempts must be > 0")

    end_idx = _compute_episode_end_indices(tout)
    # Sample base indices then expand with k goals each, then trim to batch_size.
    base_n = int(math.ceil(float(batch_size) / float(her_k_per_transition)))
    idx_base = rng.integers(0, len(obs), size=base_n, endpoint=False)

    xs_obs: List[np.ndarray] = []
    xs_goal: List[np.ndarray] = []
    xs_act: List[np.ndarray] = []
    xs_next_obs: List[np.ndarray] = []
    xs_rew: List[float] = []
    xs_done: List[float] = []

    for i in idx_base:
        i = int(i)
        done = bool(term[i] or tout[i] or (i + 1) >= len(obs))
        next_obs = obs[i] if done else obs[i + 1]
        for _ in range(int(her_k_per_transition)):
            g_idx = _sample_goal_index_for_transition(
                rng=rng,
                obs=obs,
                end_idx=end_idx,
                i=i,
                geom_p=float(geom_p),
                min_k=int(min_k),
                max_k=int(max_k),
                min_distance=float(min_distance),
                attempts=int(max(1, her_k_per_transition * future_sample_attempts)),
            )
            goal_xy = obs[int(g_idx), :2].astype(np.float32)
            dist_next = float(np.linalg.norm(next_obs[:2] - goal_xy))

            if reward_mode == "sparse":
                reward = float(sparse_success_value) if dist_next <= float(goal_success_threshold) else 0.0
            elif reward_mode == "shaped":
                reward = -float(dist_next) * float(shaped_reward_scale)
                if dist_next <= float(goal_success_threshold):
                    reward += float(shaped_success_bonus)
            else:
                raise ValueError(f"unknown reward_mode: {reward_mode}")

            xs_obs.append(obs[i].copy())
            xs_goal.append(goal_xy.copy())
            xs_act.append(act[i].copy())
            xs_next_obs.append(next_obs.copy())
            xs_rew.append(float(reward))
            xs_done.append(float(done))

    # Trim to batch_size.
    if len(xs_obs) > batch_size:
        xs_obs = xs_obs[:batch_size]
        xs_goal = xs_goal[:batch_size]
        xs_act = xs_act[:batch_size]
        xs_next_obs = xs_next_obs[:batch_size]
        xs_rew = xs_rew[:batch_size]
        xs_done = xs_done[:batch_size]

    return {
        "obs": np.asarray(xs_obs, dtype=np.float32),
        "goal": np.asarray(xs_goal, dtype=np.float32),
        "act": np.asarray(xs_act, dtype=np.float32),
        "next_obs": np.asarray(xs_next_obs, dtype=np.float32),
        "rew": np.asarray(xs_rew, dtype=np.float32).reshape(-1, 1),
        "done": np.asarray(xs_done, dtype=np.float32).reshape(-1, 1),
    }


class SACAgent:
    def __init__(
        self,
        *,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        lr: float,
        gamma: float,
        tau: float,
        policy_update_every: int,
        target_update_every: int,
        auto_alpha: bool,
        init_alpha: float,
        target_entropy: float,
        device: torch.device,
        grad_clip: float,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_update_every = max(1, int(policy_update_every))
        self.target_update_every = max(1, int(target_update_every))
        self.grad_clip = float(grad_clip)

        self.actor = SquashedGaussianActor(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.q1 = QNetwork(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
        self.q2 = QNetwork(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
        self.q1_targ = QNetwork(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
        self.q2_targ = QNetwork(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr))
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=float(lr))

        self.auto_alpha = bool(auto_alpha)
        if self.auto_alpha:
            self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(max(1e-8, float(init_alpha))), device=device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(lr))
            self.target_entropy = float(target_entropy)
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.target_entropy = float("nan")
            self._alpha_fixed = float(init_alpha)

        self.update_step = 0

    def alpha(self) -> torch.Tensor:
        if self.auto_alpha:
            assert self.log_alpha is not None
            return torch.exp(self.log_alpha)
        return torch.tensor(self._alpha_fixed, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_xy: np.ndarray, *, deterministic: bool) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        g = torch.as_tensor(goal_xy, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            a = self.actor.act_deterministic(o, g)
        else:
            a, _, _ = self.actor.sample(o, g)
        return a.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def update(self, batch: Mapping[str, np.ndarray]) -> Dict[str, float]:
        self.update_step += 1
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(batch["goal"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_act, next_logp, _ = self.actor.sample(next_obs, goal)
            q1n = self.q1_targ(next_obs, goal, next_act)
            q2n = self.q2_targ(next_obs, goal, next_act)
            qn = torch.min(q1n, q2n)
            target = rew + (1.0 - done) * self.gamma * (qn - self.alpha() * next_logp)

        q1v = self.q1(obs, goal, act)
        q2v = self.q2(obs, goal, act)
        q_loss = F.mse_loss(q1v, target) + F.mse_loss(q2v, target)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=self.grad_clip)
        self.q_opt.step()

        actor_loss = float("nan")
        alpha_loss = float("nan")
        alpha_val = float(self.alpha().detach().cpu().item())
        logp_mean = float("nan")

        if self.update_step % self.policy_update_every == 0:
            a_pi, logp, _ = self.actor.sample(obs, goal)
            q1_pi = self.q1(obs, goal, a_pi)
            q2_pi = self.q2(obs, goal, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss_t = (self.alpha() * logp - q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss_t.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
            self.actor_opt.step()

            actor_loss = float(actor_loss_t.detach().cpu().item())
            logp_mean = float(logp.detach().mean().cpu().item())

            if self.auto_alpha:
                assert self.log_alpha is not None and self.alpha_opt is not None
                alpha_loss_t = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss_t.backward()
                self.alpha_opt.step()
                alpha_loss = float(alpha_loss_t.detach().cpu().item())
                alpha_val = float(self.alpha().detach().cpu().item())

        if self.update_step % self.target_update_every == 0:
            _soft_update(self.q1_targ, self.q1, tau=self.tau)
            _soft_update(self.q2_targ, self.q2, tau=self.tau)

        return {
            "q_loss": float(q_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss),
            "alpha": float(alpha_val),
            "alpha_loss": float(alpha_loss),
            "logp_mean": float(logp_mean),
        }


@torch.no_grad()
def rollout_sac_policy_to_goal(
    agent: SACAgent,
    rollout_env: gym.Env,
    *,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rollout_horizon: int,
    rollout_mode: str,
    decision_every_n_steps: int,
    maze_arr: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    obs = dprobe.reset_rollout_start(rollout_env, start_xy=start_xy)
    min_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
    final_goal_dist = min_goal_dist
    rollout_wall_hits = int(dprobe.count_wall_hits_qpos_frame(maze_arr, obs[:2]))

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
            cached_action = agent.act(obs, goal_xy, deterministic=True)
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
def evaluate_goal_progress_sac(
    agent: SACAgent,
    *,
    env_name: str,
    query_pairs: List[Tuple[np.ndarray, np.ndarray]],
    rollout_horizon: int,
    success_prefix_horizons: Sequence[int],
    n_samples: int,
    goal_success_threshold: float,
    rollout_mode: str,
    rollout_replan_every_n_steps: int,
    maze_arr: np.ndarray | None,
) -> Dict[str, float]:
    # Provide the same columns as diffuser for shared plotting/monitoring.
    line_dev: List[float] = []
    path_ratio: List[float] = []
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
    rollout_wall_hits: List[int] = []

    rollout_env = gym.make(env_name)
    dt = float(getattr(rollout_env.unwrapped, "dt", 1.0))

    for qid, (start_xy, goal_xy) in enumerate(query_pairs):
        for _sid in range(int(n_samples)):
            rollout_xy, rollout_actions, obs_full, min_goal_dist, final_goal_dist, wall_hits, _ = rollout_sac_policy_to_goal(
                agent=agent,
                rollout_env=rollout_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                rollout_horizon=rollout_horizon,
                rollout_mode=rollout_mode,
                decision_every_n_steps=rollout_replan_every_n_steps,
                maze_arr=maze_arr,
            )
            obs_traj = obs_full[:-1]
            act_traj = rollout_actions
            xy = obs_traj[:, :2]
            s = dprobe.straightness_metrics(xy=xy, start_xy=start_xy, goal_xy=goal_xy)
            b = dprobe.boundary_jump_ratios(xy)
            c = dprobe.transition_compatibility_metrics(
                observations=obs_traj,
                actions=act_traj,
                goal_xy=goal_xy,
                dt=dt,
                goal_success_threshold=goal_success_threshold,
            )

            line_dev.append(float(s["mean_line_deviation"]))
            path_ratio.append(float(s["path_over_direct"]))
            start_jump_ratios.append(float(b["start_jump_ratio"]))
            end_jump_ratios.append(float(b["end_jump_ratio"]))
            state_velocity_l2_means.append(float(c["state_velocity_l2_mean"]))
            state_velocity_rel_means.append(float(c["state_velocity_rel_mean"]))
            velocity_action_l2_means.append(float(c["velocity_action_l2_mean"]))
            velocity_action_rel_means.append(float(c["velocity_action_rel_mean"]))

            prefix_stats = dprobe.rollout_prefix_distance_stats(
                rollout_xy=rollout_xy,
                goal_xy=goal_xy,
                prefix_horizons=success_prefix_horizons,
            )
            for h in success_prefix_horizons:
                prefix_min_goal_dist, prefix_final_goal_dist = prefix_stats[int(h)]
                hit = bool(prefix_min_goal_dist <= goal_success_threshold)
                rollout_successes_by_prefix[int(h)].append(float(hit))
                rollout_min_goal_dist_by_prefix[int(h)].append(prefix_min_goal_dist)
                rollout_final_goal_dist_by_prefix[int(h)].append(prefix_final_goal_dist)
                if hit:
                    query_success_any_by_prefix[int(h)][qid] = True

            rollout_successes.append(float(min_goal_dist <= goal_success_threshold))
            rollout_min_goal_distances.append(float(min_goal_dist))
            rollout_final_goal_errors.append(float(final_goal_dist))
            rollout_wall_hits.append(int(wall_hits))

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
        # No imagined-trajectory concept for policy rollouts; keep NaNs for schema compatibility.
        "imagined_goal_success_rate": float("nan"),
        "imagined_goal_error_mean": float("nan"),
        "imagined_pregoal_success_rate": float("nan"),
        "imagined_pregoal_error_mean": float("nan"),
        "imagined_line_deviation_mean": float(np.mean(line_dev)) if line_dev else float("nan"),
        "imagined_path_over_direct_mean": float(np.mean(path_ratio)) if path_ratio else float("nan"),
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
        "imagined_in_wall_points_mean": float("nan"),
        "rollout_in_wall_points_mean": float(np.mean(rollout_wall_hits)) if rollout_wall_hits else float("nan"),
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
    agent: SACAgent,
    *,
    env_name: str,
    replay_observations: np.ndarray,
    replay_timeouts: np.ndarray,
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
    success_threshold_flags: Dict[float, List[float]] = {float(thr): [] for thr in planning_success_thresholds}

    accepted_episodes = 0
    accepted_transitions = 0
    attempted_episodes = 0
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
                "Online collection exceeded max attempts while trying to gather enough accepted data. "
                f"accepted_eps={accepted_episodes} accepted_transitions={accepted_transitions} "
                f"attempted_eps={attempted_episodes} rejected_short={rejected_short_episodes} "
                f"min_len={min_len} transition_budget={transition_budget}"
            )

        start_xy, goal_xy, goal_k, sampled_goal_dist = dprobe.sample_geometric_start_goal_pair(
            observations=replay_observations,
            timeouts=replay_timeouts,
            rng=rng,
            geom_p=float(goal_geom_p),
            min_k=int(goal_geom_min_k),
            max_k=int(goal_geom_max_k),
            min_distance=float(goal_min_distance),
        )
        obs = dprobe.reset_rollout_start(env, start_xy=start_xy)
        initial_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = initial_goal_dist
        final_goal_dist = initial_goal_dist
        wall_hits = int(dprobe.count_wall_hits_qpos_frame(maze_arr, obs[:2]))

        ep_obs: List[np.ndarray] = []
        ep_act: List[np.ndarray] = []
        ep_rew: List[float] = []
        ep_term: List[bool] = []
        ep_tout: List[bool] = []
        ep_len = 0
        cached_action: np.ndarray | None = None
        decision_count = 0

        for t in range(int(episode_len)):
            if cached_action is None or (t % stride == 0):
                cached_action = agent.act(obs, goal_xy, deterministic=False)
                decision_count += 1
            action = np.clip(cached_action, act_low, act_high).astype(np.float32)

            next_obs, reward_env, done_env, _ = dprobe.safe_step(env, action)
            dist = float(np.linalg.norm(next_obs[:2] - goal_xy))
            min_goal_dist = min(min_goal_dist, dist)
            final_goal_dist = dist
            wall_hits += int(dprobe.count_wall_hits_qpos_frame(maze_arr, next_obs[:2]))

            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(float(reward_env))
            ep_term.append(bool(done_env))

            hit_goal = bool(early_terminate_on_success and (dist <= float(early_terminate_threshold)))
            is_timeout = bool((t == int(episode_len) - 1) or done_env or hit_goal)
            ep_tout.append(is_timeout)

            obs = next_obs
            ep_len += 1
            if done_env or hit_goal:
                break

        if ep_len > 0:
            ep_tout[-1] = True

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
                ep_obs = ep_obs[:remaining]
                ep_act = ep_act[:remaining]
                ep_rew = ep_rew[:remaining]
                ep_term = ep_term[:remaining]
                ep_tout = ep_tout[:remaining]
                ep_tout[-1] = True
                ep_len = remaining

                # Recompute distance stats on the accepted prefix.
                # Conservative: only use final obs distance; min distance may be underestimated.
                final_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))

        observations.extend(ep_obs)
        actions.extend(ep_act)
        rewards.extend(ep_rew)
        terminals.extend(ep_term)
        timeouts.extend(ep_tout)

        accepted_episodes += 1
        accepted_transitions += int(ep_len)

        episode_lengths.append(int(ep_len))
        goal_distances.append(float(sampled_goal_dist))
        goal_ks.append(int(goal_k))
        decisions_per_episode.append(int(decision_count))
        rollout_min_goal_dist.append(float(min_goal_dist))
        rollout_final_goal_dist.append(float(final_goal_dist))
        rollout_wall_hits.append(int(wall_hits))
        initial_goal_distances.append(float(initial_goal_dist))
        if initial_goal_dist > 1e-8:
            reduction_ratio = float((initial_goal_dist - final_goal_dist) / initial_goal_dist)
        else:
            reduction_ratio = float(final_goal_dist <= 1e-6)
        final_distance_reduction_ratios.append(reduction_ratio)
        success_rel_reduction_flags.append(float(reduction_ratio >= float(planning_success_rel_reduction)))
        for thr in planning_success_thresholds:
            success_threshold_flags[float(thr)].append(float(final_goal_dist <= float(thr)))

    env.close()

    dataset = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "timeouts": np.asarray(timeouts, dtype=np.bool_),
    }
    stats: Dict[str, float] = {
        "episodes": int(accepted_episodes),
        "transitions": int(accepted_transitions),
        "attempted_episodes": int(attempted_episodes),
        "rejected_short_episodes": int(rejected_short_episodes),
        "episode_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "sampled_goal_distance_mean": float(np.mean(goal_distances)) if goal_distances else 0.0,
        "sampled_goal_k_mean": float(np.mean(goal_ks)) if goal_ks else 0.0,
        "decisions_per_episode_mean": float(np.mean(decisions_per_episode)) if decisions_per_episode else 0.0,
        "rollout_min_goal_dist_mean": float(np.mean(rollout_min_goal_dist)) if rollout_min_goal_dist else 0.0,
        "rollout_final_goal_dist_mean": float(np.mean(rollout_final_goal_dist)) if rollout_final_goal_dist else 0.0,
        "rollout_wall_hits_mean": float(np.mean(rollout_wall_hits)) if rollout_wall_hits else 0.0,
        "planning_success_rate_final_rel090": float(np.mean(success_rel_reduction_flags)) if success_rel_reduction_flags else 0.0,
    }
    for thr in planning_success_thresholds:
        key = f"planning_success_rate_final_t{dprobe.threshold_tag(float(thr))}"
        stats[key] = float(np.mean(success_threshold_flags[float(thr)])) if success_threshold_flags[float(thr)] else 0.0
    return dataset, stats


def main() -> None:
    cfg = parse_args()
    dprobe.set_seed(cfg.seed)
    if cfg.online_replan_every_n_steps <= 0:
        raise ValueError("--online_replan_every_n_steps must be > 0")
    if cfg.eval_rollout_replan_every_n_steps <= 0:
        raise ValueError("--eval_rollout_replan_every_n_steps must be > 0")
    if cfg.online_goal_geom_min_k <= 0:
        raise ValueError("--online_goal_geom_min_k must be > 0")
    if cfg.online_goal_geom_max_k < cfg.online_goal_geom_min_k:
        raise ValueError("--online_goal_geom_max_k must be >= --online_goal_geom_min_k")
    if not (0.0 < cfg.online_goal_geom_p <= 1.0):
        raise ValueError("--online_goal_geom_p must be in (0, 1]")
    if cfg.goal_success_threshold <= 0.0:
        raise ValueError("--goal_success_threshold must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if cfg.her_k_per_transition <= 0:
        raise ValueError("--her_k_per_transition must be > 0")
    if cfg.reward_mode not in {"sparse", "shaped"}:
        raise ValueError("--reward_mode must be sparse|shaped")

    logdir = dprobe.make_logdir(cfg)
    with open(logdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    print(f"[setup] logdir={logdir}")

    device = torch.device(cfg.device)
    maze_arr = dprobe.load_maze_arr_from_env(cfg.env)

    # Collect offline replay with random actions (same as diffuser/gcbc scripts).
    raw_dataset, _action_low_scaled, _action_high_scaled, collection_stats = dprobe.collect_random_dataset(
        env_name=cfg.env,
        n_episodes=cfg.n_episodes,
        episode_len=cfg.episode_len,
        action_scale=cfg.action_scale,
        seed=cfg.seed,
        corridor_aware_data=cfg.corridor_aware_data,
        corridor_max_resamples=cfg.corridor_max_resamples,
    )

    # Get true env action bounds for SAC scaling.
    tmp_env = gym.make(cfg.env)
    action_low = np.asarray(tmp_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(tmp_env.action_space.high, dtype=np.float32)
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    act_dim = int(np.prod(tmp_env.action_space.shape))
    tmp_env.close()

    hidden_dims = _parse_hidden_dims(cfg.sac_hidden_dims)
    target_entropy = float(cfg.sac_target_entropy)
    if not np.isfinite(target_entropy):
        target_entropy = -float(act_dim)

    agent = SACAgent(
        obs_dim=obs_dim,
        goal_dim=2,
        action_dim=act_dim,
        hidden_dims=hidden_dims,
        log_std_min=cfg.sac_log_std_min,
        log_std_max=cfg.sac_log_std_max,
        action_low=action_low,
        action_high=action_high,
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_update_every=cfg.policy_update_every,
        target_update_every=cfg.target_update_every,
        auto_alpha=cfg.sac_auto_alpha,
        init_alpha=cfg.sac_alpha,
        target_entropy=target_entropy,
        device=device,
        grad_clip=cfg.grad_clip,
    )

    metrics_rows: List[Dict[str, float]] = []
    progress_rows: List[Dict[str, float]] = []
    online_collection_rows: List[Dict[str, float]] = []

    # Eval query bank.
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

    eval_success_prefix_horizons = dprobe.parse_int_list(cfg.eval_success_prefix_horizons, "--eval_success_prefix_horizons")
    online_planning_success_thresholds = dprobe.parse_float_list(cfg.online_planning_success_thresholds, "--online_planning_success_thresholds")

    global_step = 0
    last_eval_query_pairs = query_pairs_for_step(step=0)

    def run_updates(num_steps: int, phase: str) -> None:
        nonlocal global_step, last_eval_query_pairs, raw_dataset
        rng = np.random.default_rng(cfg.seed + 1337 + max(1, global_step))
        for _ in range(int(num_steps)):
            global_step += 1
            batch = _build_her_sac_batch(
                observations=raw_dataset["observations"],
                actions=raw_dataset["actions"],
                terminals=raw_dataset["terminals"],
                timeouts=raw_dataset["timeouts"],
                batch_size=int(cfg.batch_size),
                goal_success_threshold=float(cfg.goal_success_threshold),
                reward_mode=str(cfg.reward_mode),
                shaped_reward_scale=float(cfg.shaped_reward_scale),
                shaped_success_bonus=float(cfg.shaped_success_bonus),
                sparse_success_value=float(cfg.sparse_success_value),
                geom_p=float(cfg.online_goal_geom_p),
                min_k=int(cfg.online_goal_geom_min_k),
                max_k=int(cfg.online_goal_geom_max_k),
                min_distance=float(cfg.online_goal_min_distance),
                her_k_per_transition=int(cfg.her_k_per_transition),
                future_sample_attempts=int(cfg.her_future_sample_attempts),
                rng=rng,
            )
            info = agent.update(batch)
            metrics_rows.append(
                {
                    "step": int(global_step),
                    "phase": phase,
                    **{k: float(v) for k, v in info.items()},
                }
            )

            if global_step == 1 or global_step % 200 == 0:
                print(
                    f"[train] phase={phase} step={global_step:5d} "
                    f"q_loss={info['q_loss']:.5f} "
                    f"actor_loss={info['actor_loss']:.5f} "
                    f"alpha={info['alpha']:.4f} "
                    f"logp_mean={info['logp_mean']:.3f}"
                )

            if cfg.save_checkpoint_every > 0 and global_step % cfg.save_checkpoint_every == 0:
                ckpt = {
                    "step": int(global_step),
                    "phase": phase,
                    "actor": agent.actor.state_dict(),
                    "q1": agent.q1.state_dict(),
                    "q2": agent.q2.state_dict(),
                    "q1_targ": agent.q1_targ.state_dict(),
                    "q2_targ": agent.q2_targ.state_dict(),
                    "actor_opt": agent.actor_opt.state_dict(),
                    "q_opt": agent.q_opt.state_dict(),
                    "auto_alpha": bool(agent.auto_alpha),
                    "log_alpha": agent.log_alpha.detach().cpu().item() if agent.log_alpha is not None else float("nan"),
                    "config": asdict(cfg),
                }
                torch.save(ckpt, logdir / f"checkpoint_step{global_step}.pt")

            if cfg.eval_goal_every > 0 and global_step % cfg.eval_goal_every == 0:
                eval_query_pairs = query_pairs_for_step(step=global_step)
                last_eval_query_pairs = eval_query_pairs
                progress = evaluate_goal_progress_sac(
                    agent=agent,
                    env_name=cfg.env,
                    query_pairs=eval_query_pairs,
                    rollout_horizon=int(cfg.eval_rollout_horizon),
                    success_prefix_horizons=eval_success_prefix_horizons,
                    n_samples=int(cfg.query_batch_size),
                    goal_success_threshold=float(cfg.goal_success_threshold),
                    rollout_mode=str(cfg.eval_rollout_mode),
                    rollout_replan_every_n_steps=int(cfg.eval_rollout_replan_every_n_steps),
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
                    f"rollout_success@{short_h}={rollout_short:.3f} "
                    f"rollout_success@{long_h}={rollout_long:.3f} "
                    f"({rollout_long_count}/{int(progress['eval_num_trajectories'])}) "
                    f"goal_cov_query@{long_h}={cov_query_long:.3f} "
                    f"goal_cov_cell@{long_h}={cov_cell_long:.3f} "
                    f"rollout_goal_err={progress['rollout_final_goal_error_mean']:.3f} "
                    f"rollout_mode={progress['eval_rollout_mode']} "
                    f"query_pairs={int(progress['eval_query_pairs'])} "
                    f"roll_wall={progress['rollout_in_wall_points_mean']:.2f} "
                    f"state_vel_rel={progress['state_velocity_rel_mean']:.3f} "
                    f"vel_act_rel={progress['velocity_action_rel_mean']:.3f}"
                )

    run_updates(num_steps=cfg.train_steps, phase="offline_init")

    if cfg.online_self_improve and cfg.online_rounds > 0:
        online_min_accepted_episode_len = int(cfg.online_min_accepted_episode_len) if cfg.online_min_accepted_episode_len > 0 else int(cfg.horizon)
        for r in range(int(cfg.online_rounds)):
            round_idx = r + 1
            print(
                f"[online] round={round_idx}/{cfg.online_rounds} "
                f"collect_transition_budget_per_round={cfg.online_collect_transition_budget_per_round} "
                f"collect_eps_per_round={cfg.online_collect_episodes_per_round} "
                f"collect_episode_len={cfg.online_collect_episode_len} "
                f"decision_every={cfg.online_replan_every_n_steps}"
            )
            additions, stats = collect_policy_dataset(
                agent=agent,
                env_name=cfg.env,
                replay_observations=raw_dataset["observations"],
                replay_timeouts=raw_dataset["timeouts"],
                n_episodes=int(cfg.online_collect_episodes_per_round),
                episode_len=int(cfg.online_collect_episode_len),
                transition_budget=int(cfg.online_collect_transition_budget_per_round),
                decision_every_n_steps=int(cfg.online_replan_every_n_steps),
                goal_geom_p=float(cfg.online_goal_geom_p),
                goal_geom_min_k=int(cfg.online_goal_geom_min_k),
                goal_geom_max_k=int(cfg.online_goal_geom_max_k),
                goal_min_distance=float(cfg.online_goal_min_distance),
                seed=int(cfg.seed + 2021 + round_idx),
                maze_arr=maze_arr,
                planning_success_thresholds=online_planning_success_thresholds,
                planning_success_rel_reduction=float(cfg.online_planning_success_rel_reduction),
                early_terminate_on_success=bool(cfg.online_early_terminate_on_success),
                early_terminate_threshold=float(cfg.online_early_terminate_threshold),
                min_accepted_episode_len=int(online_min_accepted_episode_len),
            )
            raw_dataset = dprobe.merge_replay_datasets(raw_dataset, additions)
            replay_transitions = int(len(raw_dataset["observations"]))
            replay_episodes = int(dprobe.count_episodes_from_timeouts(raw_dataset["timeouts"]))

            row = {
                "round": int(round_idx),
                "phase": f"online_round_{round_idx}",
                "replay_transitions": int(replay_transitions),
                "replay_episodes": int(replay_episodes),
                **{k: float(v) for k, v in stats.items()},
            }
            online_collection_rows.append(row)
            try:
                pd.DataFrame(online_collection_rows).to_csv(logdir / "online_collection.csv", index=False)
            except Exception as e:
                print(f"[warn] failed to flush online_collection.csv: {e}")
            print(
                "[online-replay] "
                f"round={round_idx} replay_transitions={replay_transitions} replay_episodes={replay_episodes} "
                f"accepted_transitions={int(stats.get('transitions', 0))} accepted_eps={int(stats.get('episodes', 0))}"
            )

            run_updates(num_steps=cfg.online_train_steps_per_round, phase=f"online_round_{round_idx}")
    else:
        print("[online] disabled or --online_rounds <= 0, skipping online rounds.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(logdir / "metrics.csv", index=False)

    progress_df = pd.DataFrame(progress_rows)
    if len(progress_df) > 0:
        progress_df.to_csv(logdir / "progress_metrics.csv", index=False)
        dprobe.plot_progress(progress_df, out_path=logdir / "goal_progress.png")

    if len(online_collection_rows) > 0:
        pd.DataFrame(online_collection_rows).to_csv(logdir / "online_collection.csv", index=False)

    checkpoint = {
        "step": int(global_step),
        "phase": "final",
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_targ": agent.q1_targ.state_dict(),
        "q2_targ": agent.q2_targ.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "q_opt": agent.q_opt.state_dict(),
        "auto_alpha": bool(agent.auto_alpha),
        "log_alpha": agent.log_alpha.detach().cpu().item() if agent.log_alpha is not None else float("nan"),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, logdir / "checkpoint_last.pt")

    # Final query table + plots (reuse the same helper).
    final_query_pairs = last_eval_query_pairs if len(last_eval_query_pairs) > 0 else query_pairs_for_step(step=global_step)
    query_rows: List[Dict[str, float]] = []
    query_rollout_env = gym.make(cfg.env)
    for qid, (start_xy, goal_xy) in enumerate(final_query_pairs):
        for sid in range(int(cfg.query_batch_size)):
            rollout_xy, rollout_actions, obs_full, rollout_min_goal_dist, rollout_final_goal_dist, rollout_wall_hits, _ = rollout_sac_policy_to_goal(
                agent=agent,
                rollout_env=query_rollout_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                rollout_horizon=int(cfg.eval_rollout_horizon),
                rollout_mode=str(cfg.eval_rollout_mode),
                decision_every_n_steps=int(cfg.eval_rollout_replan_every_n_steps),
                maze_arr=maze_arr,
            )
            obs_traj = obs_full[:-1]
            xy = obs_traj[:, :2]
            m = dprobe.straightness_metrics(xy=xy, start_xy=start_xy, goal_xy=goal_xy)
            b = dprobe.boundary_jump_ratios(xy)
            wall_hits_imag = int(dprobe.count_wall_hits_qpos_frame(maze_arr, xy))
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
                    "imagined_in_wall_points": int(wall_hits_imag),
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
        "method": "sac_her",
        "reward_mode": str(cfg.reward_mode),
        "logdir": str(logdir),
        "train_steps_total": int(global_step),
        "online_self_improve": bool(cfg.online_self_improve),
        "online_rounds": int(cfg.online_rounds),
        "online_replan_every_n_steps": int(cfg.online_replan_every_n_steps),
        "dataset_transitions": int(len(raw_dataset["observations"])),
        "dataset_episodes": int(dprobe.count_episodes_from_timeouts(raw_dataset["timeouts"])),
        "dataset_collection_stats": collection_stats,
        "her_k_per_transition": int(cfg.her_k_per_transition),
        "final_q_loss": float(metrics_df["q_loss"].iloc[-1]) if len(metrics_df) else float("nan"),
        "final_actor_loss": float(metrics_df["actor_loss"].iloc[-1]) if len(metrics_df) else float("nan"),
        "final_alpha": float(metrics_df["alpha"].iloc[-1]) if len(metrics_df) else float("nan"),
        "eval_query_mode": cfg.query_mode,
        "eval_query_pairs_per_step": int(cfg.num_eval_queries if cfg.query_mode == "diverse" else len(query_bank)),
        "eval_query_bank_size": int(len(query_bank)),
        "eval_rollout_mode": cfg.eval_rollout_mode,
        "eval_rollout_replan_every_n_steps": int(cfg.eval_rollout_replan_every_n_steps),
        "eval_rollout_horizon": int(cfg.eval_rollout_horizon),
        "eval_success_prefix_horizons": [int(h) for h in eval_success_prefix_horizons],
        "online_planning_success_thresholds": [float(thr) for thr in online_planning_success_thresholds],
        "online_planning_success_rel_reduction": float(cfg.online_planning_success_rel_reduction),
        "query_path_over_direct_mean": float(query_df["path_over_direct"].mean()) if len(query_df) else float("nan"),
        "query_mean_line_deviation_mean": float(query_df["mean_line_deviation"].mean()) if len(query_df) else float("nan"),
        "query_rollout_min_goal_distance_mean": float(query_df["rollout_min_goal_distance"].mean()) if len(query_df) else float("nan"),
        "query_rollout_final_goal_error_mean": float(query_df["rollout_final_goal_error"].mean()) if len(query_df) else float("nan"),
        "query_imagined_in_wall_points_mean": float(query_df["imagined_in_wall_points"].mean()) if len(query_df) else float("nan"),
        "query_rollout_in_wall_points_mean": float(query_df["rollout_in_wall_points"].mean()) if len(query_df) else float("nan"),
        "progress_last": progress_rows[-1] if progress_rows else {},
        "online_collection_last": online_collection_rows[-1] if online_collection_rows else {},
    }
    with open(logdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n[done] artifacts:")
    for rel in [
        "config.json",
        "metrics.csv",
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
        if rel == "goal_progress.png" and len(progress_df) == 0:
            continue
        print(f"  - {logdir / rel}")
    print("[done] summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
