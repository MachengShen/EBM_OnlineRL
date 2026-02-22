#!/usr/bin/env python3
"""
Locomotion Collector Study: Diffuser warmstart vs SAC-scratch vs GCBC.

Three conditions:
  diffuser_warmstart_sac  – Collect N episodes with pretrained value-guided Diffuser
                            (receding-horizon, replan every H steps); warm-start a SAC
                            agent on those transitions; continue online SAC training.
  sac_scratch             – Train SAC from scratch (random warm-up, same total env-step
                            budget as diffuser_warmstart_sac).
  gcbc_diffuser           – Same Diffuser collection; train GCBC (supervised BC); no
                            online RL.

Pretrained models used (third_party/diffuser/logs/pretrained/{env}/):
  diffusion/defaults_H32_T20
  values/defaults_H32_T20_d0.997

Scale / guide parameters from config/locomotion.py per environment:
  hopper*          scale=0.0001, t_stopgrad=4
  walker2d*        scale=0.001,  t_stopgrad=4  (same as halfcheetah in config)
  halfcheetah*     scale=0.001,  t_stopgrad=4

Output: <out_dir>/locomotion_collector_results.csv
        columns: condition, env, seed, grad_step, norm_score, env_steps_total
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSER_LOCO = REPO_ROOT / "third_party" / "diffuser"
MUJOCO_BIN = "/root/.mujoco/mujoco210/bin"

_ld = os.environ.get("LD_LIBRARY_PATH", "")
if MUJOCO_BIN not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{_ld}:{MUJOCO_BIN}" if _ld else MUJOCO_BIN
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

# Insert locomotion diffuser BEFORE gym imports so the right d4rl is used
sys.path.insert(0, str(DIFFUSER_LOCO))

# ---------------------------------------------------------------------------
# NullRenderer patch  – prevents GLEW crash from MjRenderContextOffscreen
# ---------------------------------------------------------------------------
import diffuser.utils.rendering as _rmod  # noqa: E402


class _NullRenderer:
    def __init__(self, *a, **kw): pass
    def composite(self, *a, **kw): return None
    def renders(self, *a, **kw): return None
    def render(self, *a, **kw): return None


_rmod.MuJoCoRenderer = _NullRenderer

import diffuser.sampling as dsampling  # noqa: E402
import diffuser.utils as dutils  # noqa: E402
import d4rl  # noqa: E402, F401
import gym  # noqa: E402


# ---------------------------------------------------------------------------
# Per-environment tuning from config/locomotion.py
# ---------------------------------------------------------------------------
_ENV_SCALE: Dict[str, float] = {
    "hopper": 0.0001,
    "walker2d": 0.001,
    "halfcheetah": 0.001,
}
_ENV_TSTOPGRAD: Dict[str, int] = {
    "hopper": 4,
    "walker2d": 4,
    "halfcheetah": 4,
}

PRETRAINED_BASE = str(DIFFUSER_LOCO / "logs" / "pretrained")
DIFFUSION_SUBPATH = "diffusion/defaults_H32_T20"
VALUE_SUBPATH = "values/defaults_H32_T20_d0.997"
PLAN_HORIZON = 32   # matches pretrained H32


def _env_family(env_name: str) -> str:
    for k in _ENV_SCALE:
        if k in env_name:
            return k
    raise ValueError(f"Unknown env family for: {env_name}")


# ---------------------------------------------------------------------------
# Diffuser Collector
# ---------------------------------------------------------------------------

def build_diffuser_policy(env_name: str, device: str) -> dsampling.GuidedPolicy:
    """Load pretrained Diffuser + Value function, return a GuidedPolicy."""
    fam = _env_family(env_name)
    diff_exp = dutils.load_diffusion(
        PRETRAINED_BASE, env_name, DIFFUSION_SUBPATH,
        epoch="latest", device=device,
    )
    val_exp = dutils.load_diffusion(
        PRETRAINED_BASE, env_name, VALUE_SUBPATH,
        epoch="latest", device=device,
    )
    guide = dsampling.ValueGuide(val_exp.ema)
    policy = dsampling.GuidedPolicy(
        guide=guide,
        diffusion_model=diff_exp.ema,
        normalizer=diff_exp.dataset.normalizer,
        preprocess_fns=[],
        sample_fn=dsampling.n_step_guided_p_sample,
        n_guide_steps=2,
        scale=_ENV_SCALE[fam],
        t_stopgrad=_ENV_TSTOPGRAD[fam],
        scale_grad_by_std=True,
    )
    return policy


def collect_diffuser_episodes(
    env: gym.Env,
    policy: dsampling.GuidedPolicy,
    n_episodes: int,
    max_steps: int,
    replan_every: int,
    rng: np.random.RandomState,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], float]:
    """
    Collect n_episodes using the GuidedPolicy with receding-horizon replanning.

    Key efficiency: replan every `replan_every` steps (== plan horizon H).
    Between replans, execute the pre-planned actions from the stored trajectory.
    This reduces policy calls from max_steps to max_steps/replan_every.

    Returns (obs_list, act_list, rew_list, next_obs_list, done_list, mean_norm_score).
    Each element is a list of arrays, one per episode.
    """
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    scores = []
    for ep in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
        planned_actions: Optional[np.ndarray] = None  # [H, action_dim]
        plan_idx = replan_every  # force replan at t=0

        for t in range(max_steps):
            # Replan when we've exhausted the current plan
            if plan_idx >= replan_every:
                conditions = {0: np.asarray(obs, dtype=np.float32)}
                _, trajectories = policy(conditions, batch_size=1, verbose=False)
                # trajectories.actions: [batch_size, H, action_dim] — unnormalized
                planned_actions = trajectories.actions[0]  # [H, action_dim]
                plan_idx = 0

            action = planned_actions[plan_idx]
            plan_idx += 1

            next_obs, reward, done, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(float(reward))
            ep_nobs.append(next_obs.copy())
            ep_done.append(float(done))
            total_r += reward
            obs = next_obs
            if done:
                break

        score = float(env.get_normalized_score(total_r))
        scores.append(score)
        obs_l.append(np.array(ep_obs, dtype=np.float32))
        act_l.append(np.array(ep_act, dtype=np.float32))
        rew_l.append(np.array(ep_rew, dtype=np.float32))
        nobs_l.append(np.array(ep_nobs, dtype=np.float32))
        done_l.append(np.array(ep_done, dtype=np.float32))
        print(f"  [Diffuser collect] ep {ep+1}/{n_episodes}  steps={len(ep_obs)}  score={score:.2f}", flush=True)

    return obs_l, act_l, rew_l, nobs_l, done_l, float(np.mean(scores))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular numpy replay buffer for continuous-action RL."""
    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ):
        N = len(obs)
        for i in range(N):
            idx = self.ptr % self.max_size
            self.obs[idx] = obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_obs[idx] = next_obs[i]
            self.dones[idx] = dones[i]
            self.ptr += 1
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idxs], dtype=torch.float32).to(device)
        return (
            to_t(self.obs),
            to_t(self.actions),
            to_t(self.rewards),
            to_t(self.next_obs),
            to_t(self.dones),
        )


# ---------------------------------------------------------------------------
# SAC Actor / Critic
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential:
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class SACActorLoco(nn.Module):
    """Squashed Gaussian actor (obs only, no goal)."""
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, 2 * action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str, deterministic: bool = False) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        if deterministic:
            out = self.net(t)
            mean, _ = torch.chunk(out, 2, dim=-1)
            action = torch.tanh(mean).squeeze(0)
        else:
            action, _ = self.forward(t)
            action = action.squeeze(0)
        return action.cpu().numpy()


class DoubleQCriticLoco(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = _build_mlp(in_dim, hidden_dims, 1)
        self.q2 = _build_mlp(in_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class SACAgentLoco:
    """Standard SAC (no goal-conditioning) for locomotion environments."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        auto_alpha: bool = True,
        alpha: float = 0.2,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        self.actor = SACActorLoco(obs_dim, action_dim, hidden_dims).to(device)
        self.critic = DoubleQCriticLoco(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target = DoubleQCriticLoco(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(action_dim)
        if auto_alpha:
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = float(torch.exp(self.log_alpha).item())
        else:
            self.log_alpha = None
            self.alpha = alpha

    def update(self, replay: ReplayBuffer, batch_size: int = 256) -> Dict[str, float]:
        obs, actions, rewards, next_obs, dones = replay.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1.0 - dones) * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_pi = self.actor(obs)
        q_new = self.critic.min_q(obs, new_actions)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(torch.exp(self.log_alpha).item())

        # Soft update target critic
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(p.data * self.tau)

        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss), "alpha": self.alpha}


# ---------------------------------------------------------------------------
# GCBC Policy
# ---------------------------------------------------------------------------

class GCBCPolicyLoco(nn.Module):
    """Deterministic BC policy: obs -> action (tanh squashed)."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        return self.forward(t).squeeze(0).cpu().numpy()


def train_gcbc(
    replay: ReplayBuffer,
    obs_dim: int,
    action_dim: int,
    device: str,
    n_steps: int = 10_000,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> GCBCPolicyLoco:
    policy = GCBCPolicyLoco(obs_dim, action_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        obs, actions, _, _, _ = replay.sample(batch_size, device)
        pred = policy(obs)
        loss = F.mse_loss(pred, actions)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sac(
    env: gym.Env,
    actor: SACActorLoco,
    device: str,
    n_episodes: int = 10,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> float:
    scores = []
    for _ in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        for _ in range(max_steps):
            action = actor.act(obs, device, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            total_r += reward
            if done:
                break
        scores.append(float(env.get_normalized_score(total_r)))
    return float(np.mean(scores))


def evaluate_gcbc(
    env: gym.Env,
    policy: GCBCPolicyLoco,
    device: str,
    n_episodes: int = 10,
    max_steps: int = 1000,
) -> float:
    scores = []
    for _ in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        for _ in range(max_steps):
            action = policy.act(obs, device)
            obs, reward, done, _ = env.step(action)
            total_r += reward
            if done:
                break
        scores.append(float(env.get_normalized_score(total_r)))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Online Diffuser training utilities
# ---------------------------------------------------------------------------

import diffuser.models.temporal as _temporal_mod   # noqa: E402
import diffuser.models.diffusion as _diffusion_mod  # noqa: E402
from diffuser.utils.training import EMA             # noqa: E402
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer  # noqa: E402


def _env_dims(env_name: str) -> Tuple[int, int]:
    """Return (obs_dim, action_dim) for a gym env without keeping it open."""
    e = gym.make(env_name)
    obs_dim = e.observation_space.shape[0]
    act_dim = e.action_space.shape[0]
    e.close()
    return obs_dim, act_dim


def build_diffuser_from_scratch(
    obs_dim: int,
    action_dim: int,
    device: str,
    horizon: int = 32,
    n_diffusion_steps: int = 20,
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
) -> Tuple[Any, Any, Any, torch.optim.Optimizer]:
    """
    Construct a randomly initialized Diffuser model for locomotion.

    Returns (diffusion_model, ema_model, ema_helper, optimizer).
    """
    transition_dim = obs_dim + action_dim
    model = _temporal_mod.TemporalUnet(
        horizon=horizon,
        transition_dim=transition_dim,
        cond_dim=obs_dim,
        dim_mults=dim_mults,
        attention=False,
    ).to(device)

    diffusion = _diffusion_mod.GaussianDiffusion(
        model=model,
        horizon=horizon,
        observation_dim=obs_dim,
        action_dim=action_dim,
        n_timesteps=n_diffusion_steps,
        loss_type="l2",
        clip_denoised=False,
        predict_epsilon=False,
        action_weight=10,
        loss_discount=1,
        loss_weights=None,
    ).to(device)

    ema_model = copy.deepcopy(diffusion)
    ema_helper = EMA(0.995)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

    return diffusion, ema_model, ema_helper, optimizer


def episodes_to_sequence_dataset(
    obs_list: List[np.ndarray],
    act_list: List[np.ndarray],
    obs_dim: int,
    action_dim: int,
    horizon: int = 32,
    max_path_length: int = 1000,
) -> torch.utils.data.Dataset:
    """
    Convert collected episodes to a Dataset yielding (trajectories, conditions)
    batches compatible with GaussianDiffusion.loss().

    Each episode is segmented into overlapping windows of length `horizon`.
    Normalization uses per-dim min/max from the collected data.
    """
    # Build per-episode arrays padded to max_path_length
    n_episodes = len(obs_list)
    observations = np.zeros((n_episodes, max_path_length, obs_dim), dtype=np.float32)
    actions = np.zeros((n_episodes, max_path_length, action_dim), dtype=np.float32)
    path_lengths = np.zeros(n_episodes, dtype=np.int64)

    for i in range(n_episodes):
        L = min(len(obs_list[i]), max_path_length)
        observations[i, :L] = obs_list[i][:L]
        actions[i, :L] = act_list[i][:L]
        path_lengths[i] = L

    # Compute normalizer from all real data
    all_obs = np.concatenate([observations[i, :path_lengths[i]] for i in range(n_episodes)], axis=0)
    all_act = np.concatenate([actions[i, :path_lengths[i]] for i in range(n_episodes)], axis=0)

    obs_min = all_obs.min(axis=0)
    obs_max = all_obs.max(axis=0)
    act_min = all_act.min(axis=0)
    act_max = all_act.max(axis=0)

    # Avoid division by zero
    obs_range = np.maximum(obs_max - obs_min, 1e-6)
    act_range = np.maximum(act_max - act_min, 1e-6)

    # Normalize to [-1, 1]
    normed_obs = np.zeros_like(observations)
    normed_act = np.zeros_like(actions)
    for i in range(n_episodes):
        L = path_lengths[i]
        normed_obs[i, :L] = 2.0 * (observations[i, :L] - obs_min) / obs_range - 1.0
        normed_act[i, :L] = 2.0 * (actions[i, :L] - act_min) / act_range - 1.0

    # Build indices: (episode_idx, start, end) for trajectory windows
    # Use padding=True semantics: allow windows that extend past episode end
    # (padding is zeros, which is already the default in our arrays)
    indices = []
    for i in range(n_episodes):
        L = path_lengths[i]
        if L < 2:
            continue
        # Allow windows starting up to L-1 (the array is zero-padded beyond L)
        max_start = min(L - 1, max_path_length - horizon)
        for start in range(max(max_start, 1)):
            indices.append((i, start, start + horizon))

    indices = np.array(indices, dtype=np.int64)

    class _LocoSeqDataset(torch.utils.data.Dataset):
        """Lightweight dataset matching Diffuser Batch format."""
        def __init__(self):
            self.indices = indices
            self.normed_obs = normed_obs
            self.normed_act = normed_act
            self.normalizer_info = {
                "obs_min": obs_min, "obs_max": obs_max,
                "act_min": act_min, "act_max": act_max,
            }

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            ep, start, end = self.indices[idx]
            obs_chunk = self.normed_obs[ep, start:end]   # (H, obs_dim)
            act_chunk = self.normed_act[ep, start:end]   # (H, action_dim)
            # Diffuser format: trajectories = [actions, observations] concatenated
            trajectories = np.concatenate([act_chunk, obs_chunk], axis=-1)  # (H, action_dim+obs_dim)
            conditions = {0: obs_chunk[0]}  # condition on first observation
            return trajectories, conditions

    return _LocoSeqDataset()


class _LocoNormalizer:
    """Minimal normalizer for Diffuser policy inference with online-collected data."""
    def __init__(self, obs_min, obs_max, act_min, act_max):
        self.obs_min = obs_min.astype(np.float32)
        self.obs_max = obs_max.astype(np.float32)
        self.act_min = act_min.astype(np.float32)
        self.act_max = act_max.astype(np.float32)
        self.obs_range = np.maximum(self.obs_max - self.obs_min, 1e-6)
        self.act_range = np.maximum(self.act_max - self.act_min, 1e-6)

    def normalize(self, x, key):
        if key == "observations":
            return 2.0 * (x - self.obs_min) / self.obs_range - 1.0
        elif key == "actions":
            return 2.0 * (x - self.act_min) / self.act_range - 1.0
        return x

    def unnormalize(self, x, key):
        if key == "observations":
            return (x + 1.0) / 2.0 * self.obs_range + self.obs_min
        elif key == "actions":
            return (x + 1.0) / 2.0 * self.act_range + self.act_min
        return x


def train_diffuser_steps(
    diffusion: Any,
    ema_model: Any,
    ema_helper: EMA,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    n_steps: int,
    device: str,
    grad_clip: float = 1.0,
    ema_update_every: int = 10,
) -> float:
    """Run n_steps of diffusion training. Returns mean loss."""
    from itertools import cycle as _cycle
    train_iter = _cycle(dataloader)
    losses = []
    diffusion.train()
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        trajectories, conditions = next(train_iter)
        trajectories = trajectories.float().to(device)
        conditions = {k: v.float().to(device) for k, v in conditions.items()}
        loss, _ = diffusion.loss(trajectories, conditions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
        optimizer.step()
        if step % ema_update_every == 0:
            ema_helper.update_model_average(ema_model, diffusion)
        losses.append(loss.item())
    diffusion.eval()
    return float(np.mean(losses)) if losses else 0.0


def evaluate_diffuser_loco(
    env: gym.Env,
    ema_model: Any,
    normalizer: _LocoNormalizer,
    n_episodes: int,
    max_steps: int = 1000,
    replan_every: int = 32,
) -> float:
    """Evaluate a trained Diffuser model (no value guide, pure diffusion planning)."""
    ema_model.eval()
    scores = []
    for _ in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        planned_actions = None
        plan_idx = replan_every

        for t in range(max_steps):
            if plan_idx >= replan_every:
                obs_normed = normalizer.normalize(
                    np.asarray(obs, dtype=np.float32), "observations"
                )
                conditions = {0: torch.tensor(obs_normed, dtype=torch.float32).unsqueeze(0)}
                device = next(ema_model.parameters()).device
                conditions = {k: v.to(device) for k, v in conditions.items()}
                with torch.no_grad():
                    # Sample trajectory from the diffusion model
                    sample_result = ema_model.conditional_sample(conditions)
                    # sample_result is a Sample namedtuple; [0] = trajectories (1, H, transition_dim)
                    samples = sample_result[0]
                    actions_normed = samples[0, :, :ema_model.action_dim].cpu().numpy()
                planned_actions = normalizer.unnormalize(actions_normed, "actions")
                plan_idx = 0

            action = np.clip(planned_actions[plan_idx], env.action_space.low, env.action_space.high)
            plan_idx += 1
            obs, reward, done, _ = env.step(action)
            total_r += reward
            if done:
                break

        scores.append(float(env.get_normalized_score(total_r)))
    return float(np.mean(scores))


def collect_diffuser_episodes_online(
    env: gym.Env,
    ema_model: Any,
    normalizer: _LocoNormalizer,
    n_episodes: int,
    max_steps: int,
    replan_every: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], float]:
    """Collect episodes using trained Diffuser (no value guide)."""
    ema_model.eval()
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    scores = []
    for ep in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
        planned_actions = None
        plan_idx = replan_every

        for t in range(max_steps):
            if plan_idx >= replan_every:
                obs_normed = normalizer.normalize(
                    np.asarray(obs, dtype=np.float32), "observations"
                )
                conditions = {0: torch.tensor(obs_normed, dtype=torch.float32).unsqueeze(0)}
                device = next(ema_model.parameters()).device
                conditions = {k: v.to(device) for k, v in conditions.items()}
                with torch.no_grad():
                    sample_result = ema_model.conditional_sample(conditions)
                    samples = sample_result[0]
                    actions_normed = samples[0, :, :ema_model.action_dim].cpu().numpy()
                planned_actions = normalizer.unnormalize(actions_normed, "actions")
                plan_idx = 0

            action = np.clip(planned_actions[plan_idx], env.action_space.low, env.action_space.high)
            plan_idx += 1
            next_obs, reward, done, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(float(reward))
            ep_nobs.append(next_obs.copy())
            ep_done.append(float(done))
            total_r += reward
            obs = next_obs
            if done:
                break

        score = float(env.get_normalized_score(total_r))
        scores.append(score)
        obs_l.append(np.array(ep_obs, dtype=np.float32))
        act_l.append(np.array(ep_act, dtype=np.float32))
        rew_l.append(np.array(ep_rew, dtype=np.float32))
        nobs_l.append(np.array(ep_nobs, dtype=np.float32))
        done_l.append(np.array(ep_done, dtype=np.float32))

    return obs_l, act_l, rew_l, nobs_l, done_l, float(np.mean(scores))


# ---------------------------------------------------------------------------
# Online SAC training loop
# ---------------------------------------------------------------------------

def run_online_sac(
    env: gym.Env,
    eval_env: gym.Env,
    agent: SACAgentLoco,
    replay: ReplayBuffer,
    n_online_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    min_replay_before_train: int,
    train_steps_per_step: int,
    eval_every_episodes: int,
    n_eval_episodes: int,
    device: str,
    rng: np.random.RandomState,
    label: str = "",
    env_steps_offset: int = 0,
) -> List[Dict]:
    """Interleave episode collection + SAC updates. Return list of eval records."""
    records: List[Dict] = []
    grad_steps = 0
    env_steps_total = env_steps_offset

    for ep in range(n_online_episodes):
        obs = env.reset()
        ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
        for t in range(max_episode_steps):
            if replay.size < min_replay_before_train:
                action = env.action_space.sample()
            else:
                action = agent.actor.act(obs, device)
            next_obs, reward, done, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action)
            ep_rew.append(float(reward))
            ep_nobs.append(next_obs.copy())
            ep_done.append(float(done))
            obs = next_obs
            env_steps_total += 1
            if done:
                break

        replay.add_episode(
            np.array(ep_obs), np.array(ep_act), np.array(ep_rew),
            np.array(ep_nobs), np.array(ep_done),
        )

        # Train
        steps_this_ep = len(ep_obs)
        for _ in range(steps_this_ep * train_steps_per_step):
            if replay.size >= min_replay_before_train:
                agent.update(replay, batch_size)
                grad_steps += 1

        # Evaluate
        if (ep + 1) % eval_every_episodes == 0:
            score = evaluate_sac(eval_env, agent.actor, device, n_eval_episodes)
            print(f"  [{label}] ep={ep+1}/{n_online_episodes}  grad_steps={grad_steps}  score={score:.2f}", flush=True)
            records.append({
                "grad_step": grad_steps,
                "env_steps_total": env_steps_total,
                "norm_score": score,
            })

    return records


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_condition_diffuser_warmstart_sac(
    env_name: str,
    seed: int,
    device: str,
    n_diffuser_episodes: int,
    n_online_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    train_steps_per_step: int,
    eval_every_episodes: int,
    n_eval_episodes: int,
    replan_every: int,
) -> List[Dict]:
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[diffuser_warmstart_sac] env={env_name} seed={seed} collecting {n_diffuser_episodes} Diffuser episodes...", flush=True)
    t0 = time.time()
    policy = build_diffuser_policy(env_name, device)
    obs_l, act_l, rew_l, nobs_l, done_l, collect_score = collect_diffuser_episodes(
        env, policy, n_diffuser_episodes, max_episode_steps, replan_every, rng
    )
    print(f"  Collection done in {time.time()-t0:.1f}s  mean_score={collect_score:.2f}", flush=True)
    del policy  # free GPU memory

    replay = ReplayBuffer(obs_dim, action_dim)
    for i in range(len(obs_l)):
        replay.add_episode(obs_l[i], act_l[i], rew_l[i], nobs_l[i], done_l[i])

    # Evaluate Diffuser collection quality directly
    eval_env_tmp = gym.make(env_name)
    eval_env_tmp.seed(seed + 20000)
    collect_eval_score = collect_score  # already computed above

    agent = SACAgentLoco(obs_dim, action_dim, device)
    env_steps_offset = sum(len(o) for o in obs_l)

    # Record baseline before any SAC training (just diffuser data quality)
    records = [{
        "grad_step": 0,
        "env_steps_total": env_steps_offset,
        "norm_score": collect_eval_score,
    }]

    online_records = run_online_sac(
        env, eval_env, agent, replay,
        n_online_episodes=n_online_episodes,
        max_episode_steps=max_episode_steps,
        batch_size=batch_size,
        min_replay_before_train=batch_size,
        train_steps_per_step=train_steps_per_step,
        eval_every_episodes=eval_every_episodes,
        n_eval_episodes=n_eval_episodes,
        device=device,
        rng=rng,
        label="diffuser_warmstart_sac",
        env_steps_offset=env_steps_offset,
    )
    records.extend(online_records)
    return records


def run_condition_sac_scratch(
    env_name: str,
    seed: int,
    device: str,
    n_warmup_episodes: int,
    n_online_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    train_steps_per_step: int,
    eval_every_episodes: int,
    n_eval_episodes: int,
) -> List[Dict]:
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay = ReplayBuffer(obs_dim, action_dim)
    agent = SACAgentLoco(obs_dim, action_dim, device)

    total_episodes = n_warmup_episodes + n_online_episodes
    print(f"[sac_scratch] env={env_name} seed={seed} running {total_episodes} episodes...", flush=True)

    records = run_online_sac(
        env, eval_env, agent, replay,
        n_online_episodes=total_episodes,
        max_episode_steps=max_episode_steps,
        batch_size=batch_size,
        min_replay_before_train=batch_size,
        train_steps_per_step=train_steps_per_step,
        eval_every_episodes=eval_every_episodes,
        n_eval_episodes=n_eval_episodes,
        device=device,
        rng=rng,
        label="sac_scratch",
    )
    return records


def run_condition_gcbc_diffuser(
    env_name: str,
    seed: int,
    device: str,
    n_diffuser_episodes: int,
    max_episode_steps: int,
    n_gcbc_steps: int,
    batch_size: int,
    n_eval_episodes: int,
    replan_every: int,
) -> List[Dict]:
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[gcbc_diffuser] env={env_name} seed={seed} collecting {n_diffuser_episodes} Diffuser episodes...", flush=True)
    t0 = time.time()
    policy = build_diffuser_policy(env_name, device)
    obs_l, act_l, rew_l, nobs_l, done_l, collect_score = collect_diffuser_episodes(
        env, policy, n_diffuser_episodes, max_episode_steps, replan_every, rng
    )
    print(f"  Collection done in {time.time()-t0:.1f}s  mean_score={collect_score:.2f}", flush=True)
    del policy

    replay = ReplayBuffer(obs_dim, action_dim)
    for i in range(len(obs_l)):
        replay.add_episode(obs_l[i], act_l[i], rew_l[i], nobs_l[i], done_l[i])

    print(f"  Training GCBC for {n_gcbc_steps} steps on {replay.size} transitions...", flush=True)
    gcbc = train_gcbc(replay, obs_dim, action_dim, device, n_steps=n_gcbc_steps, batch_size=batch_size)

    score = evaluate_gcbc(eval_env, gcbc, device, n_eval_episodes)
    env_steps = sum(len(o) for o in obs_l)
    print(f"  GCBC score={score:.2f}", flush=True)

    return [{"grad_step": n_gcbc_steps, "env_steps_total": env_steps, "norm_score": score}]


def run_condition_diffuser_online(
    env_name: str,
    seed: int,
    device: str,
    n_online_episodes: int,
    max_episode_steps: int,
    diffuser_train_steps_per_round: int,
    diffuser_train_batch_size: int,
    collect_per_round: int,
    eval_every_episodes: int,
    n_eval_episodes: int,
    replan_every: int,
) -> List[Dict]:
    """
    Pure Diffuser online loop: Diffuser collects AND learns from scratch.

    Every `collect_per_round` episodes, rebuild the dataset from ALL collected
    episodes and retrain the diffusion model for `diffuser_train_steps_per_round`
    gradient steps.  Evaluate the EMA model periodically.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[diffuser_online] env={env_name} seed={seed} building random Diffuser...", flush=True)
    diffusion, ema_model, ema_helper, optimizer = build_diffuser_from_scratch(
        obs_dim, action_dim, device, horizon=PLAN_HORIZON,
    )

    # Accumulate all collected episodes
    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    records: List[Dict] = []
    grad_steps = 0
    env_steps_total = 0
    normalizer: Optional[_LocoNormalizer] = None

    for ep in range(n_online_episodes):
        # ----- Collect one episode -----
        if normalizer is None:
            # No trained model yet: collect random episode
            obs = env.reset()
            ep_obs, ep_act, ep_rew = [], [], []
            for t in range(max_episode_steps):
                action = env.action_space.sample()
                next_obs, reward, done, _ = env.step(action)
                ep_obs.append(obs.copy())
                ep_act.append(action.copy())
                ep_rew.append(float(reward))
                obs = next_obs
                env_steps_total += 1
                if done:
                    break
            score = float(env.get_normalized_score(sum(ep_rew)))
            all_obs.append(np.array(ep_obs, dtype=np.float32))
            all_act.append(np.array(ep_act, dtype=np.float32))
        else:
            # Collect with current Diffuser model
            o_l, a_l, r_l, _, _, score = collect_diffuser_episodes_online(
                env, ema_model, normalizer, 1, max_episode_steps, replan_every
            )
            all_obs.extend(o_l)
            all_act.extend(a_l)
            env_steps_total += sum(len(o) for o in o_l)

        # ----- Retrain Diffuser periodically -----
        if (ep + 1) % collect_per_round == 0 and len(all_obs) >= 2:
            print(f"  [diffuser_online] ep={ep+1} retraining on {len(all_obs)} episodes "
                  f"({sum(len(o) for o in all_obs)} transitions)...", flush=True)

            dataset = episodes_to_sequence_dataset(
                all_obs, all_act, obs_dim, action_dim,
                horizon=PLAN_HORIZON, max_path_length=max_episode_steps,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=diffuser_train_batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )

            mean_loss = train_diffuser_steps(
                diffusion, ema_model, ema_helper, optimizer, loader,
                n_steps=diffuser_train_steps_per_round, device=device,
            )
            grad_steps += diffuser_train_steps_per_round

            # Update normalizer from dataset
            info = dataset.normalizer_info
            normalizer = _LocoNormalizer(
                info["obs_min"], info["obs_max"],
                info["act_min"], info["act_max"],
            )
            print(f"    loss={mean_loss:.4f}  grad_steps={grad_steps}", flush=True)

        # ----- Evaluate periodically -----
        if (ep + 1) % eval_every_episodes == 0:
            if normalizer is not None:
                eval_score = evaluate_diffuser_loco(
                    eval_env, ema_model, normalizer, n_eval_episodes,
                    max_steps=max_episode_steps, replan_every=replan_every,
                )
            else:
                eval_score = 0.0
            print(f"  [diffuser_online] ep={ep+1}/{n_online_episodes}  "
                  f"grad_steps={grad_steps}  score={eval_score:.2f}", flush=True)
            records.append({
                "grad_step": grad_steps,
                "env_steps_total": env_steps_total,
                "norm_score": eval_score,
            })

    return records


def run_condition_diffuser_collects_sac_learns(
    env_name: str,
    seed: int,
    device: str,
    n_online_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    train_steps_per_step: int,
    diffuser_train_steps_per_round: int,
    diffuser_train_batch_size: int,
    collect_per_round: int,
    eval_every_episodes: int,
    n_eval_episodes: int,
    replan_every: int,
) -> List[Dict]:
    """
    Diffuser collects (with online improvement) → SAC learns from same data.

    The Diffuser does online self-improvement (collect + retrain), while
    simultaneously an SAC agent trains on the Diffuser-collected data.
    This tests Diffuser as a collector vs SAC as a learner.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[diff_collects_sac_learns] env={env_name} seed={seed} building random Diffuser + SAC...", flush=True)
    diffusion, ema_model, ema_helper, optimizer = build_diffuser_from_scratch(
        obs_dim, action_dim, device, horizon=PLAN_HORIZON,
    )
    agent = SACAgentLoco(obs_dim, action_dim, device)
    replay = ReplayBuffer(obs_dim, action_dim)

    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    records: List[Dict] = []
    grad_steps = 0
    env_steps_total = 0
    normalizer: Optional[_LocoNormalizer] = None

    for ep in range(n_online_episodes):
        # ----- Collect one episode with Diffuser (or random initially) -----
        if normalizer is None:
            obs = env.reset()
            ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
            for t in range(max_episode_steps):
                action = env.action_space.sample()
                next_obs, reward, done, _ = env.step(action)
                ep_obs.append(obs.copy())
                ep_act.append(action.copy())
                ep_rew.append(float(reward))
                ep_nobs.append(next_obs.copy())
                ep_done.append(float(done))
                obs = next_obs
                env_steps_total += 1
                if done:
                    break
            all_obs.append(np.array(ep_obs, dtype=np.float32))
            all_act.append(np.array(ep_act, dtype=np.float32))
            replay.add_episode(
                np.array(ep_obs), np.array(ep_act), np.array(ep_rew),
                np.array(ep_nobs), np.array(ep_done),
            )
        else:
            o_l, a_l, r_l, no_l, d_l, _ = collect_diffuser_episodes_online(
                env, ema_model, normalizer, 1, max_episode_steps, replan_every
            )
            all_obs.extend(o_l)
            all_act.extend(a_l)
            for i in range(len(o_l)):
                replay.add_episode(o_l[i], a_l[i], r_l[i], no_l[i], d_l[i])
                env_steps_total += len(o_l[i])

        # ----- Train SAC on collected data -----
        ep_len = len(all_obs[-1])
        for _ in range(ep_len * train_steps_per_step):
            if replay.size >= batch_size:
                agent.update(replay, batch_size)
                grad_steps += 1

        # ----- Retrain Diffuser periodically -----
        if (ep + 1) % collect_per_round == 0 and len(all_obs) >= 2:
            dataset = episodes_to_sequence_dataset(
                all_obs, all_act, obs_dim, action_dim,
                horizon=PLAN_HORIZON, max_path_length=max_episode_steps,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=diffuser_train_batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            train_diffuser_steps(
                diffusion, ema_model, ema_helper, optimizer, loader,
                n_steps=diffuser_train_steps_per_round, device=device,
            )
            info = dataset.normalizer_info
            normalizer = _LocoNormalizer(
                info["obs_min"], info["obs_max"],
                info["act_min"], info["act_max"],
            )

        # ----- Evaluate SAC periodically -----
        if (ep + 1) % eval_every_episodes == 0:
            sac_score = evaluate_sac(eval_env, agent.actor, device, n_eval_episodes)
            print(f"  [diff_collects_sac_learns] ep={ep+1}/{n_online_episodes}  "
                  f"grad_steps={grad_steps}  sac_score={sac_score:.2f}", flush=True)
            records.append({
                "grad_step": grad_steps,
                "env_steps_total": env_steps_total,
                "norm_score": sac_score,
            })

    return records


def run_condition_sac_collects_diffuser_learns(
    env_name: str,
    seed: int,
    device: str,
    n_online_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    train_steps_per_step: int,
    diffuser_train_steps_per_round: int,
    diffuser_train_batch_size: int,
    collect_per_round: int,
    eval_every_episodes: int,
    n_eval_episodes: int,
    replan_every: int,
) -> List[Dict]:
    """
    SAC collects (with online improvement) → Diffuser learns from SAC's data.

    SAC runs its normal online loop (collect + train), while simultaneously
    a Diffuser trains on the SAC-collected data.
    This tests SAC as a collector vs Diffuser as a learner.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[sac_collects_diff_learns] env={env_name} seed={seed} building SAC + random Diffuser...", flush=True)
    agent = SACAgentLoco(obs_dim, action_dim, device)
    replay = ReplayBuffer(obs_dim, action_dim)
    diffusion, ema_model, ema_helper, diff_optimizer = build_diffuser_from_scratch(
        obs_dim, action_dim, device, horizon=PLAN_HORIZON,
    )

    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    records: List[Dict] = []
    grad_steps = 0
    env_steps_total = 0
    normalizer: Optional[_LocoNormalizer] = None

    for ep in range(n_online_episodes):
        # ----- Collect one episode with SAC -----
        obs = env.reset()
        ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
        for t in range(max_episode_steps):
            if replay.size < batch_size:
                action = env.action_space.sample()
            else:
                action = agent.actor.act(obs, device)
            next_obs, reward, done, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(float(reward))
            ep_nobs.append(next_obs.copy())
            ep_done.append(float(done))
            obs = next_obs
            env_steps_total += 1
            if done:
                break

        replay.add_episode(
            np.array(ep_obs), np.array(ep_act), np.array(ep_rew),
            np.array(ep_nobs), np.array(ep_done),
        )
        all_obs.append(np.array(ep_obs, dtype=np.float32))
        all_act.append(np.array(ep_act, dtype=np.float32))

        # ----- Train SAC -----
        ep_len = len(ep_obs)
        for _ in range(ep_len * train_steps_per_step):
            if replay.size >= batch_size:
                agent.update(replay, batch_size)
                grad_steps += 1

        # ----- Retrain Diffuser periodically on SAC's data -----
        if (ep + 1) % collect_per_round == 0 and len(all_obs) >= 2:
            dataset = episodes_to_sequence_dataset(
                all_obs, all_act, obs_dim, action_dim,
                horizon=PLAN_HORIZON, max_path_length=max_episode_steps,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=diffuser_train_batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            train_diffuser_steps(
                diffusion, ema_model, ema_helper, diff_optimizer, loader,
                n_steps=diffuser_train_steps_per_round, device=device,
            )
            info = dataset.normalizer_info
            normalizer = _LocoNormalizer(
                info["obs_min"], info["obs_max"],
                info["act_min"], info["act_max"],
            )

        # ----- Evaluate Diffuser periodically -----
        if (ep + 1) % eval_every_episodes == 0:
            if normalizer is not None:
                diff_score = evaluate_diffuser_loco(
                    eval_env, ema_model, normalizer, n_eval_episodes,
                    max_steps=max_episode_steps, replan_every=replan_every,
                )
            else:
                diff_score = 0.0
            print(f"  [sac_collects_diff_learns] ep={ep+1}/{n_online_episodes}  "
                  f"grad_steps={grad_steps}  diff_score={diff_score:.2f}", flush=True)
            records.append({
                "grad_step": grad_steps,
                "env_steps_total": env_steps_total,
                "norm_score": diff_score,
            })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Config:
    envs: str = "hopper-medium-expert-v2"          # comma-separated
    conditions: str = "diffuser_warmstart_sac,sac_scratch,gcbc_diffuser"
    seeds: str = "0,1,2"
    device: str = "cuda:0"
    out_dir: str = ""
    n_diffuser_episodes: int = 5       # Diffuser collection episodes
    n_online_episodes: int = 100       # Online SAC episodes after warmup
    max_episode_steps: int = 1000
    batch_size: int = 256
    train_steps_per_step: int = 2      # SAC grad steps per env step
    eval_every_episodes: int = 20
    n_eval_episodes: int = 5
    replan_every: int = 32             # Replan every H steps (=horizon)
    n_gcbc_steps: int = 10_000
    # Diffuser online training params
    diffuser_train_steps_per_round: int = 1000  # Diffuser grad steps per retrain round
    diffuser_train_batch_size: int = 32         # Diffuser training batch size
    collect_per_round: int = 10                 # Collect N episodes before retraining Diffuser


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Locomotion collector study")
    p.add_argument("--envs", type=str, default=Config.envs)
    p.add_argument("--conditions", type=str, default=Config.conditions)
    p.add_argument("--seeds", type=str, default=Config.seeds)
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--out-dir", dest="out_dir", type=str, default=Config.out_dir)
    p.add_argument("--n-diffuser-episodes", dest="n_diffuser_episodes", type=int, default=Config.n_diffuser_episodes)
    p.add_argument("--n-online-episodes", dest="n_online_episodes", type=int, default=Config.n_online_episodes)
    p.add_argument("--max-episode-steps", dest="max_episode_steps", type=int, default=Config.max_episode_steps)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
    p.add_argument("--train-steps-per-step", dest="train_steps_per_step", type=int, default=Config.train_steps_per_step)
    p.add_argument("--eval-every-episodes", dest="eval_every_episodes", type=int, default=Config.eval_every_episodes)
    p.add_argument("--n-eval-episodes", dest="n_eval_episodes", type=int, default=Config.n_eval_episodes)
    p.add_argument("--replan-every", dest="replan_every", type=int, default=Config.replan_every)
    p.add_argument("--n-gcbc-steps", dest="n_gcbc_steps", type=int, default=Config.n_gcbc_steps)
    p.add_argument("--diffuser-train-steps-per-round", dest="diffuser_train_steps_per_round", type=int, default=Config.diffuser_train_steps_per_round)
    p.add_argument("--diffuser-train-batch-size", dest="diffuser_train_batch_size", type=int, default=Config.diffuser_train_batch_size)
    p.add_argument("--collect-per-round", dest="collect_per_round", type=int, default=Config.collect_per_round)
    return Config(**vars(p.parse_args()))


def main():
    cfg = parse_args()

    envs = [e.strip() for e in cfg.envs.split(",") if e.strip()]
    conditions = [c.strip() for c in cfg.conditions.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in cfg.seeds.split(",") if s.strip()]

    out_dir = Path(cfg.out_dir) if cfg.out_dir else (
        REPO_ROOT / "runs" / "analysis" / "locomotion_collector"
        / f"loco_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Config saved to {cfg_path}", flush=True)

    csv_path = out_dir / "locomotion_collector_results.csv"
    fieldnames = ["condition", "env", "seed", "grad_step", "env_steps_total", "norm_score"]

    all_rows = []

    for env_name in envs:
        for condition in conditions:
            for seed in seeds:
                print(f"\n{'='*60}", flush=True)
                print(f"ENV={env_name}  CONDITION={condition}  SEED={seed}", flush=True)
                print(f"{'='*60}", flush=True)

                try:
                    if condition == "diffuser_warmstart_sac":
                        records = run_condition_diffuser_warmstart_sac(
                            env_name, seed, cfg.device,
                            cfg.n_diffuser_episodes, cfg.n_online_episodes,
                            cfg.max_episode_steps, cfg.batch_size,
                            cfg.train_steps_per_step,
                            cfg.eval_every_episodes, cfg.n_eval_episodes,
                            cfg.replan_every,
                        )
                    elif condition == "sac_scratch":
                        records = run_condition_sac_scratch(
                            env_name, seed, cfg.device,
                            cfg.n_diffuser_episodes,   # same warmup budget
                            cfg.n_online_episodes,
                            cfg.max_episode_steps, cfg.batch_size,
                            cfg.train_steps_per_step,
                            cfg.eval_every_episodes, cfg.n_eval_episodes,
                        )
                    elif condition == "gcbc_diffuser":
                        records = run_condition_gcbc_diffuser(
                            env_name, seed, cfg.device,
                            cfg.n_diffuser_episodes,
                            cfg.max_episode_steps, cfg.n_gcbc_steps,
                            cfg.batch_size, cfg.n_eval_episodes,
                            cfg.replan_every,
                        )
                    elif condition == "diffuser_online":
                        records = run_condition_diffuser_online(
                            env_name, seed, cfg.device,
                            cfg.n_online_episodes,
                            cfg.max_episode_steps,
                            cfg.diffuser_train_steps_per_round,
                            cfg.diffuser_train_batch_size,
                            cfg.collect_per_round,
                            cfg.eval_every_episodes, cfg.n_eval_episodes,
                            cfg.replan_every,
                        )
                    elif condition == "diffuser_collects_sac_learns":
                        records = run_condition_diffuser_collects_sac_learns(
                            env_name, seed, cfg.device,
                            cfg.n_online_episodes,
                            cfg.max_episode_steps, cfg.batch_size,
                            cfg.train_steps_per_step,
                            cfg.diffuser_train_steps_per_round,
                            cfg.diffuser_train_batch_size,
                            cfg.collect_per_round,
                            cfg.eval_every_episodes, cfg.n_eval_episodes,
                            cfg.replan_every,
                        )
                    elif condition == "sac_collects_diffuser_learns":
                        records = run_condition_sac_collects_diffuser_learns(
                            env_name, seed, cfg.device,
                            cfg.n_online_episodes,
                            cfg.max_episode_steps, cfg.batch_size,
                            cfg.train_steps_per_step,
                            cfg.diffuser_train_steps_per_round,
                            cfg.diffuser_train_batch_size,
                            cfg.collect_per_round,
                            cfg.eval_every_episodes, cfg.n_eval_episodes,
                            cfg.replan_every,
                        )
                    else:
                        raise ValueError(f"Unknown condition: {condition}")

                    for rec in records:
                        row = {"condition": condition, "env": env_name, "seed": seed, **rec}
                        all_rows.append(row)
                        print(f"  RECORD: {row}", flush=True)

                except Exception as e:
                    import traceback
                    print(f"  ERROR in {condition}/{env_name}/seed={seed}: {e}", flush=True)
                    traceback.print_exc()
                    all_rows.append({
                        "condition": condition, "env": env_name, "seed": seed,
                        "grad_step": -1, "env_steps_total": -1, "norm_score": float("nan"),
                    })

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {csv_path}", flush=True)
    print(f"Total rows: {len(all_rows)}", flush=True)

    # Quick summary
    import pandas as pd
    df = pd.read_csv(csv_path)
    print("\n=== SUMMARY (final eval per condition/env/seed) ===", flush=True)
    last = df.sort_values("grad_step").groupby(["condition", "env", "seed"]).last().reset_index()
    print(last[["condition", "env", "seed", "norm_score"]].to_string(), flush=True)
    summary = last.groupby(["condition", "env"])["norm_score"].agg(["mean", "std"]).reset_index()
    print("\n=== MEAN ± STD ===", flush=True)
    print(summary.to_string(), flush=True)

    # Save summary JSON
    summary_path = out_dir / "locomotion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(f"Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
