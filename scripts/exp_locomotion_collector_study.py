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
from typing import Dict, List, Optional, Sequence, Tuple

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
