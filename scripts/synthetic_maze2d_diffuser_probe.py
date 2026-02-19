#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import d4rl  # noqa: F401
from diffuser.datasets.sequence import GoalDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import TemporalUnet
from diffuser.utils.arrays import batch_to_device
from diffuser.utils.training import EMA


def safe_reset(env: gym.Env, seed: int | None = None) -> np.ndarray:
    try:
        out = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def safe_step(env: gym.Env, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        next_obs, reward, done, info = out
    return np.asarray(next_obs, dtype=np.float32), float(reward), bool(done), dict(info)


def cycle(dataloader: Iterable):
    while True:
        for batch in dataloader:
            yield batch


REPLAY_NPZ_KEYS = ("observations", "actions", "rewards", "terminals", "timeouts")
REPLAY_METADATA_JSON_KEY = "replay_metadata_json"


def _episode_len_stats_from_timeouts(timeouts: np.ndarray) -> Dict[str, float]:
    timeouts = np.asarray(timeouts).astype(bool).reshape(-1)
    lens: List[int] = []
    cur = 0
    for done in timeouts:
        cur += 1
        if done:
            lens.append(cur)
            cur = 0
    if cur > 0:
        lens.append(cur)
    if not lens:
        return {"episode_len_mean": 0.0, "episode_len_min": 0, "episode_len_max": 0}
    return {
        "episode_len_mean": float(np.mean(lens)),
        "episode_len_min": int(np.min(lens)),
        "episode_len_max": int(np.max(lens)),
    }


def _replay_array_digest(arr: np.ndarray, max_items: int = 8192) -> str:
    flat = np.asarray(arr).reshape(-1)
    if flat.size > max_items * 2:
        sample = np.concatenate([flat[:max_items], flat[-max_items:]], axis=0)
    else:
        sample = flat
    h = hashlib.sha256()
    h.update(np.asarray(sample).tobytes(order="C"))
    return h.hexdigest()


def replay_dataset_fingerprint(dataset: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in REPLAY_NPZ_KEYS:
        arr = np.asarray(dataset[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(_replay_array_digest(arr).encode("utf-8"))
    return h.hexdigest()[:16]


def save_replay_artifact(
    path: Path,
    dataset: Mapping[str, np.ndarray],
    action_low: np.ndarray,
    action_high: np.ndarray,
    collection_stats: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(dataset[k]) for k in REPLAY_NPZ_KEYS}
    replay_episodes = int(count_episodes_from_timeouts(payload["timeouts"]))
    replay_transitions = int(len(payload["observations"]))
    replay_meta: Dict[str, Any] = dict(metadata or {})
    replay_meta.setdefault("format", "maze2d_replay_npz_v1")
    replay_meta.setdefault("transitions", replay_transitions)
    replay_meta.setdefault("episodes", replay_episodes)
    replay_meta.setdefault("fingerprint", replay_dataset_fingerprint(payload))
    payload["action_low"] = np.asarray(action_low, dtype=np.float32)
    payload["action_high"] = np.asarray(action_high, dtype=np.float32)
    payload["collection_stats_json"] = np.asarray(json.dumps(collection_stats), dtype=np.str_)
    payload[REPLAY_METADATA_JSON_KEY] = np.asarray(json.dumps(replay_meta), dtype=np.str_)
    np.savez_compressed(str(path), **payload)
    return replay_meta


def load_replay_artifact(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as f:
        dataset = {k: np.asarray(f[k]) for k in REPLAY_NPZ_KEYS}
        action_low = np.asarray(f["action_low"], dtype=np.float32) if "action_low" in f else None
        action_high = np.asarray(f["action_high"], dtype=np.float32) if "action_high" in f else None
        stats_json = str(f["collection_stats_json"].item()) if "collection_stats_json" in f else "{}"
        meta_json = str(f[REPLAY_METADATA_JSON_KEY].item()) if REPLAY_METADATA_JSON_KEY in f else "{}"
    try:
        stats = dict(json.loads(stats_json))
    except Exception:
        stats = {}
    try:
        replay_meta = dict(json.loads(meta_json))
    except Exception:
        replay_meta = {}
    # Ensure keys expected by downstream logging exist.
    stats.setdefault("wall_rejects", 0)
    stats.setdefault("failed_steps", 0)
    if "episode_len_mean" not in stats or "episode_len_min" not in stats or "episode_len_max" not in stats:
        stats.update(_episode_len_stats_from_timeouts(dataset["timeouts"]))
    if action_low is None or action_high is None:
        raise ValueError(f"Replay file missing action_low/action_high: {path}")
    replay_meta.setdefault("format", "maze2d_replay_npz_v1")
    replay_meta.setdefault("transitions", int(len(dataset["observations"])))
    replay_meta.setdefault("episodes", int(count_episodes_from_timeouts(dataset["timeouts"])))
    replay_meta.setdefault("fingerprint", replay_dataset_fingerprint(dataset))
    return dataset, action_low, action_high, stats, replay_meta


def save_replay_npz(
    path: Path,
    dataset: Dict[str, np.ndarray],
    action_low: np.ndarray,
    action_high: np.ndarray,
    collection_stats: Dict[str, float],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    _ = save_replay_artifact(
        path=path,
        dataset=dataset,
        action_low=action_low,
        action_high=action_high,
        collection_stats=collection_stats,
        metadata=metadata,
    )


def load_replay_npz(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, float]]:
    dataset, action_low, action_high, stats, _ = load_replay_artifact(path=path)
    return dataset, action_low, action_high, stats


@dataclass
class Config:
    env: str = "maze2d-umaze-v1"
    seed: int = 0
    device: str = "cuda:0"
    logdir: str = ""
    n_episodes: int = 400
    episode_len: int = 192
    horizon: int = 64
    n_diffusion_steps: int = 64
    model_dim: int = 64
    model_dim_mults: str = "1,2,4"
    learning_rate: float = 2e-4
    train_steps: int = 4000
    batch_size: int = 128
    grad_clip: float = 1.0
    val_frac: float = 0.1
    val_every: int = 100
    val_batches: int = 20
    ema_decay: float = 0.995
    ema_update_every: int = 10
    ema_start_step: int = 500
    action_scale: float = 1.0
    corridor_aware_data: bool = False
    corridor_max_resamples: int = 200
    clip_denoised: bool = True
    predict_epsilon: bool = True
    max_path_length: int = 256
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
    eval_waypoint_mode: str = "none"  # {"none","feasible","infeasible"}
    eval_waypoint_t: int = 0  # 0 => horizon//2
    eval_waypoint_eps: float = 0.2
    wall_aware_planning: bool = True
    wall_aware_plan_samples: int = 8
    save_checkpoint_every: int = 5000
    online_self_improve: bool = False
    disable_online_collection: bool = False
    online_rounds: int = 0
    online_train_steps_per_round: int = 2000
    online_collect_episodes_per_round: int = 32
    online_collect_episode_len: int = 256
    # If >0, collect until this many *accepted* transitions are added per online round.
    # This makes the online sample budget explicit (env-step budget), which matters once
    # episodes can end early on success.
    online_collect_transition_budget_per_round: int = 0
    online_replan_every_n_steps: int = 8
    online_goal_geom_p: float = 0.08
    online_goal_geom_min_k: int = 8
    online_goal_geom_max_k: int = 96
    online_goal_min_distance: float = 0.5
    online_planning_success_thresholds: str = "0.1,0.2"
    online_planning_success_rel_reduction: float = 0.9
    # Option A (recommended): end an online-collection episode early once it reaches
    # the goal region, then reset and sample a new (start, goal) pair.
    online_early_terminate_on_success: bool = True
    online_early_terminate_threshold: float = 0.2
    # Reject (discard) online-collection episodes shorter than this many steps.
    # Use 0 to default to the diffusion horizon.
    online_min_accepted_episode_len: int = 0

    # Bootstrapping / ablation knobs:
    # - collector_weights: which weights to use for online collection when no
    #   external collector checkpoint is provided.
    # - eval_weights: which learner weights to use for evaluation and query plots.
    #   (Training always updates the online model; EMA is a tracking copy.)
    collector_weights: str = "ema"  # {"ema","online"}
    eval_weights: str = "ema"  # {"ema","online"}
    # Optional: use a frozen collector (teacher) loaded from a checkpoint.
    collector_ckpt_path: str = ""
    collector_ckpt_weights: str = "ema"  # {"ema","online"} -> checkpoint key {ema,model}
    # Optional: load/save replay snapshots for fixed-replay experiments.
    replay_import_path: str = ""
    replay_export_path: str = ""
    replay_load_npz: str = ""
    replay_save_npz: str = ""
    # If >0: after collecting this online round, snapshot replay and freeze further appends.
    fixed_replay_snapshot_round: int = 0
    fixed_replay_snapshot_npz: str = ""


class SyntheticDatasetEnv:
    """
    Minimal env-like object expected by GoalDataset/sequence_dataset.
    """

    def __init__(self, name: str, dataset: Dict[str, np.ndarray], max_episode_steps: int):
        self.name = name
        self._dataset = dataset
        self._max_episode_steps = int(max_episode_steps)
        self.max_episode_steps = int(max_episode_steps)
        self._target = np.zeros(2, dtype=np.float32)

    def get_dataset(self):
        return self._dataset

    def seed(self, seed: int | None = None):
        return


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Maze2D experiment with original Diffuser implementation: "
            "random-policy data collection, training, and start-goal trajectory probing."
        )
    )
    parser.add_argument("--env", type=str, default=Config.env)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--logdir", type=str, default=Config.logdir)
    parser.add_argument("--n_episodes", type=int, default=Config.n_episodes)
    parser.add_argument("--episode_len", type=int, default=Config.episode_len)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--n_diffusion_steps", type=int, default=Config.n_diffusion_steps)
    parser.add_argument("--model_dim", type=int, default=Config.model_dim)
    parser.add_argument("--model_dim_mults", type=str, default=Config.model_dim_mults)
    parser.add_argument("--learning_rate", type=float, default=Config.learning_rate)
    parser.add_argument("--train_steps", type=int, default=Config.train_steps)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--grad_clip", type=float, default=Config.grad_clip)
    parser.add_argument("--val_frac", type=float, default=Config.val_frac)
    parser.add_argument("--val_every", type=int, default=Config.val_every)
    parser.add_argument("--val_batches", type=int, default=Config.val_batches)
    parser.add_argument("--ema_decay", type=float, default=Config.ema_decay)
    parser.add_argument("--ema_update_every", type=int, default=Config.ema_update_every)
    parser.add_argument("--ema_start_step", type=int, default=Config.ema_start_step)
    parser.add_argument("--action_scale", type=float, default=Config.action_scale)
    parser.add_argument(
        "--corridor_aware_data",
        dest="corridor_aware_data",
        action="store_true",
        help="Reject collected transitions that land inside maze-wall cells.",
    )
    parser.add_argument(
        "--no_corridor_aware_data",
        dest="corridor_aware_data",
        action="store_false",
        help="Disable wall-aware rejection in synthetic data collection.",
    )
    parser.set_defaults(corridor_aware_data=Config.corridor_aware_data)
    parser.add_argument("--corridor_max_resamples", type=int, default=Config.corridor_max_resamples)
    parser.add_argument("--max_path_length", type=int, default=Config.max_path_length)
    parser.add_argument("--query", type=str, default=Config.query)
    parser.add_argument(
        "--query_mode",
        type=str,
        default=Config.query_mode,
        choices=["fixed", "diverse"],
        help="fixed: use --query string only; diverse: generate angularly diverse start-goal pairs from dataset observations.",
    )
    parser.add_argument("--num_eval_queries", type=int, default=Config.num_eval_queries)
    parser.add_argument("--query_bank_size", type=int, default=Config.query_bank_size)
    parser.add_argument("--query_angle_bins", type=int, default=Config.query_angle_bins)
    parser.add_argument("--query_min_distance", type=float, default=Config.query_min_distance)
    parser.add_argument(
        "--query_resample_each_eval",
        dest="query_resample_each_eval",
        action="store_true",
        help="For diverse query mode, resample a new subset of query pairs at each eval step.",
    )
    parser.add_argument(
        "--no_query_resample_each_eval",
        dest="query_resample_each_eval",
        action="store_false",
        help="For diverse query mode, reuse the same sampled query subset at every eval step.",
    )
    parser.set_defaults(query_resample_each_eval=Config.query_resample_each_eval)
    parser.add_argument("--query_resample_seed_stride", type=int, default=Config.query_resample_seed_stride)
    parser.add_argument("--query_batch_size", type=int, default=Config.query_batch_size)
    parser.add_argument("--eval_goal_every", type=int, default=Config.eval_goal_every)
    parser.add_argument("--goal_success_threshold", type=float, default=Config.goal_success_threshold)
    parser.add_argument(
        "--eval_rollout_mode",
        type=str,
        default=Config.eval_rollout_mode,
        choices=["open_loop", "receding_horizon"],
        help="How to execute eval rollouts: open-loop action playback or receding-horizon replanning.",
    )
    parser.add_argument(
        "--eval_rollout_replan_every_n_steps",
        type=int,
        default=Config.eval_rollout_replan_every_n_steps,
        help="Eval replanning cadence for receding_horizon mode; 1 means replan every step.",
    )
    parser.add_argument(
        "--eval_rollout_horizon",
        type=int,
        default=Config.eval_rollout_horizon,
        help="Env-step budget used by eval rollouts; prefix metrics are derived from this single rollout.",
    )
    parser.add_argument(
        "--eval_success_prefix_horizons",
        type=str,
        default=Config.eval_success_prefix_horizons,
        help="Comma-separated rollout prefixes for success metrics (e.g. 64,128,192,256).",
    )
    parser.add_argument(
        "--eval_waypoint_mode",
        type=str,
        default=Config.eval_waypoint_mode,
        choices=["none", "feasible", "infeasible"],
        help="Optional eval mode: add a waypoint constraint at timestep --eval_waypoint_t.",
    )
    parser.add_argument(
        "--eval_waypoint_t",
        type=int,
        default=Config.eval_waypoint_t,
        help="Waypoint timestep (0 => planning_horizon//2). Clamped to [1, planning_horizon-2].",
    )
    parser.add_argument(
        "--eval_waypoint_eps",
        type=float,
        default=Config.eval_waypoint_eps,
        help="Waypoint hit threshold in qpos xy distance.",
    )
    parser.add_argument(
        "--wall_aware_planning",
        dest="wall_aware_planning",
        action="store_true",
        help="During imagined planning, sample multiple candidates and prefer trajectories that avoid wall cells.",
    )
    parser.add_argument(
        "--no_wall_aware_planning",
        dest="wall_aware_planning",
        action="store_false",
        help="Disable wall-aware candidate selection during planning.",
    )
    parser.set_defaults(wall_aware_planning=Config.wall_aware_planning)
    parser.add_argument(
        "--wall_aware_plan_samples",
        type=int,
        default=Config.wall_aware_plan_samples,
        help="Number of imagined candidates per plan when wall-aware planning is enabled.",
    )
    parser.add_argument("--save_checkpoint_every", type=int, default=Config.save_checkpoint_every)
    parser.add_argument(
        "--online_self_improve",
        dest="online_self_improve",
        action="store_true",
        help="Enable online self-improvement rounds: collect planner rollouts into replay and continue training.",
    )
    parser.add_argument(
        "--no_online_self_improve",
        dest="online_self_improve",
        action="store_false",
        help="Disable online self-improvement rounds.",
    )
    parser.set_defaults(online_self_improve=Config.online_self_improve)
    parser.add_argument(
        "--disable_online_collection",
        dest="disable_online_collection",
        action="store_true",
        help="Keep online rounds/training but skip new env collection (fixed replay).",
    )
    parser.add_argument(
        "--enable_online_collection",
        dest="disable_online_collection",
        action="store_false",
        help="Enable normal online collection (default).",
    )
    parser.set_defaults(disable_online_collection=Config.disable_online_collection)
    parser.add_argument("--online_rounds", type=int, default=Config.online_rounds)
    parser.add_argument("--online_train_steps_per_round", type=int, default=Config.online_train_steps_per_round)
    parser.add_argument("--online_collect_episodes_per_round", type=int, default=Config.online_collect_episodes_per_round)
    parser.add_argument("--online_collect_episode_len", type=int, default=Config.online_collect_episode_len)
    parser.add_argument(
        "--online_collect_transition_budget_per_round",
        type=int,
        default=Config.online_collect_transition_budget_per_round,
        help=(
            "If >0, collect until this many accepted transitions are added per online round "
            "(env-step budget). If 0, collect a fixed number of episodes instead."
        ),
    )
    parser.add_argument(
        "--online_replan_every_n_steps",
        type=int,
        default=Config.online_replan_every_n_steps,
        help="Receding-horizon replanning cadence in env steps; 1 means replan every step.",
    )
    parser.add_argument("--online_goal_geom_p", type=float, default=Config.online_goal_geom_p)
    parser.add_argument("--online_goal_geom_min_k", type=int, default=Config.online_goal_geom_min_k)
    parser.add_argument("--online_goal_geom_max_k", type=int, default=Config.online_goal_geom_max_k)
    parser.add_argument("--online_goal_min_distance", type=float, default=Config.online_goal_min_distance)
    parser.add_argument(
        "--online_planning_success_thresholds",
        type=str,
        default=Config.online_planning_success_thresholds,
        help="Comma-separated final-distance thresholds used to mark planner success during online collection.",
    )
    parser.add_argument(
        "--online_planning_success_rel_reduction",
        type=float,
        default=Config.online_planning_success_rel_reduction,
        help="Success criterion on relative final-distance reduction vs. initial start-goal distance (e.g., 0.9 = 90%%).",
    )
    parser.add_argument(
        "--online_early_terminate_on_success",
        dest="online_early_terminate_on_success",
        action="store_true",
        help="End an online-collection episode early once within --online_early_terminate_threshold of the goal.",
    )
    parser.add_argument(
        "--no_online_early_terminate_on_success",
        dest="online_early_terminate_on_success",
        action="store_false",
        help="Disable early termination on goal success during online collection.",
    )
    parser.set_defaults(online_early_terminate_on_success=Config.online_early_terminate_on_success)
    parser.add_argument("--online_early_terminate_threshold", type=float, default=Config.online_early_terminate_threshold)
    parser.add_argument(
        "--online_min_accepted_episode_len",
        type=int,
        default=Config.online_min_accepted_episode_len,
        help="Reject (discard) online-collection episodes shorter than this many steps (0 => diffusion horizon).",
    )
    parser.add_argument(
        "--collector_weights",
        type=str,
        default=Config.collector_weights,
        choices=["ema", "online"],
        help="Which learner weights to use for online collection planning (when --collector_ckpt_path is not set).",
    )
    parser.add_argument(
        "--eval_weights",
        type=str,
        default=Config.eval_weights,
        choices=["ema", "online"],
        help="Which learner weights to use for evaluation and query-rollout plots.",
    )
    parser.add_argument(
        "--collector_ckpt_path",
        type=str,
        default=Config.collector_ckpt_path,
        help="Optional: checkpoint .pt path to use as a frozen collector (teacher) for online collection.",
    )
    parser.add_argument(
        "--collector_ckpt_weights",
        type=str,
        default=Config.collector_ckpt_weights,
        choices=["ema", "online"],
        help="If --collector_ckpt_path is set, which weights to load: ema or online.",
    )
    parser.add_argument(
        "--replay_import_path",
        type=str,
        default=Config.replay_import_path,
        help="Optional: import replay artifact (.npz) before training (preferred alias).",
    )
    parser.add_argument(
        "--replay_export_path",
        type=str,
        default=Config.replay_export_path,
        help="Optional: export replay artifact (.npz) at end of run (preferred alias).",
    )
    parser.add_argument(
        "--replay_load_npz",
        type=str,
        default=Config.replay_load_npz,
        help="Optional: load replay dataset (.npz) instead of collecting random-policy data.",
    )
    parser.add_argument(
        "--replay_save_npz",
        type=str,
        default=Config.replay_save_npz,
        help="Optional: save final replay dataset (.npz) at end of run.",
    )
    parser.add_argument(
        "--fixed_replay_snapshot_round",
        type=int,
        default=Config.fixed_replay_snapshot_round,
        help="If >0: snapshot replay after this online round and freeze further appends for remaining rounds.",
    )
    parser.add_argument(
        "--fixed_replay_snapshot_npz",
        type=str,
        default=Config.fixed_replay_snapshot_npz,
        help="Where to write the fixed-replay snapshot (.npz). Default: <logdir>/replay_snapshot_roundN.npz.",
    )
    parser.add_argument(
        "--clip_denoised",
        dest="clip_denoised",
        action="store_true",
        help="Clamp denoised trajectory to [-1, 1] during sampling (original default).",
    )
    parser.add_argument(
        "--no_clip_denoised",
        dest="clip_denoised",
        action="store_false",
        help="Disable trajectory clipping.",
    )
    parser.set_defaults(clip_denoised=Config.clip_denoised)
    parser.add_argument(
        "--predict_epsilon",
        dest="predict_epsilon",
        action="store_true",
        help="Train the diffusion model to predict epsilon.",
    )
    parser.add_argument(
        "--predict_x0",
        dest="predict_epsilon",
        action="store_false",
        help="Train diffusion model to predict x0 directly.",
    )
    parser.set_defaults(predict_epsilon=Config.predict_epsilon)
    return Config(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dim_mults(raw: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("--model_dim_mults must contain at least one integer")
    return vals


def parse_int_list(raw: str, arg_name: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{arg_name} must contain at least one integer")
    return vals


def parse_float_list(raw: str, arg_name: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{arg_name} must contain at least one float")
    return vals


def threshold_tag(value: float) -> str:
    return f"{int(round(value * 100.0)):03d}"


def parse_queries(raw: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    queries: List[Tuple[np.ndarray, np.ndarray]] = []
    chunks = [c.strip() for c in raw.split(";") if c.strip()]
    for chunk in chunks:
        try:
            start_raw, goal_raw = chunk.split(":")
            sx, sy = [float(v.strip()) for v in start_raw.split(",")]
            gx, gy = [float(v.strip()) for v in goal_raw.split(",")]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid query format '{chunk}'. Expected 'sx,sy:gx,gy'.") from exc
        queries.append(
            (
                np.asarray([sx, sy], dtype=np.float32),
                np.asarray([gx, gy], dtype=np.float32),
            )
        )
    if not queries:
        raise ValueError("No valid queries parsed from --query")
    return queries


def _unique_xy_points(points_xy: np.ndarray, decimals: int = 3) -> np.ndarray:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape [N, 2]")
    rounded = np.round(points_xy.astype(np.float32), decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return points_xy[np.sort(idx)]


def build_diverse_query_bank(
    points_xy: np.ndarray,
    bank_size: int,
    n_angle_bins: int,
    min_pair_distance: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if bank_size <= 0:
        raise ValueError("bank_size must be > 0")
    if n_angle_bins < 4:
        raise ValueError("n_angle_bins must be >= 4")

    rng = np.random.default_rng(seed)
    points = _unique_xy_points(np.asarray(points_xy, dtype=np.float32))
    if len(points) < 2:
        raise ValueError("Need at least two unique points to build query bank")

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    bin_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    angle_bins = np.digitize(angles, bin_edges[1:-1], right=False)

    by_bin: Dict[int, np.ndarray] = {
        b: np.where(angle_bins == b)[0]
        for b in range(n_angle_bins)
    }
    nonempty_bins = [b for b in range(n_angle_bins) if len(by_bin[b]) > 0]
    if len(nonempty_bins) < 2:
        raise ValueError("Not enough angular diversity in points to build diverse queries")

    bank: List[Tuple[np.ndarray, np.ndarray]] = []
    max_attempts = bank_size * 300
    attempts = 0
    while len(bank) < bank_size and attempts < max_attempts:
        attempts += 1
        anchor_bin = nonempty_bins[len(bank) % len(nonempty_bins)]
        opposite_bin = (anchor_bin + (n_angle_bins // 2)) % n_angle_bins
        if len(by_bin[opposite_bin]) == 0:
            opposite_bin = int(rng.choice(nonempty_bins))

        start_bucket = by_bin[anchor_bin] if len(by_bin[anchor_bin]) > 0 else np.arange(len(points))
        goal_bucket = by_bin[opposite_bin] if len(by_bin[opposite_bin]) > 0 else np.arange(len(points))
        s_idx = int(rng.choice(start_bucket))
        g_idx = int(rng.choice(goal_bucket))
        if s_idx == g_idx:
            continue

        start_xy = points[s_idx]
        goal_xy = points[g_idx]
        if float(np.linalg.norm(goal_xy - start_xy)) < min_pair_distance:
            continue

        bank.append((start_xy.astype(np.float32), goal_xy.astype(np.float32)))

    if len(bank) < bank_size:
        # Fallback: unconstrained random pair fill (still respecting min distance when possible).
        fill_attempts = 0
        while len(bank) < bank_size and fill_attempts < max_attempts:
            fill_attempts += 1
            s_idx = int(rng.integers(0, len(points)))
            g_idx = int(rng.integers(0, len(points)))
            if s_idx == g_idx:
                continue
            start_xy = points[s_idx]
            goal_xy = points[g_idx]
            if float(np.linalg.norm(goal_xy - start_xy)) < min_pair_distance:
                continue
            bank.append((start_xy.astype(np.float32), goal_xy.astype(np.float32)))

    if not bank:
        raise ValueError("Failed to build any query pairs")
    return bank


def select_query_pairs(
    query_bank: Sequence[Tuple[np.ndarray, np.ndarray]],
    num_queries: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")
    if len(query_bank) == 0:
        raise ValueError("query_bank cannot be empty")

    rng = np.random.default_rng(seed)
    replace = num_queries > len(query_bank)
    idx = rng.choice(len(query_bank), size=num_queries, replace=replace)
    return [query_bank[int(i)] for i in idx]


def make_logdir(cfg: Config) -> Path:
    if cfg.logdir:
        logdir = Path(cfg.logdir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = Path("runs") / "analysis" / "synth_maze2d_diffuser_probe" / stamp
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir


def parse_maze_spec(maze_spec: str) -> np.ndarray:
    lines = maze_spec.strip().split("\\")
    if not lines or not lines[0]:
        raise ValueError("Empty maze specification")
    width = len(lines)
    height = len(lines[0])
    arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        if len(lines[w]) != height:
            raise ValueError("Maze specification has inconsistent row lengths")
        for h in range(height):
            tile = lines[w][h]
            if tile == "#":
                arr[w, h] = 10  # WALL
            else:
                arr[w, h] = 11  # FREE/GOAL
    return arr


def load_maze_arr_from_env(env_name: str) -> np.ndarray | None:
    env = gym.make(env_name)
    try:
        maze_spec = getattr(env.unwrapped, "str_maze_spec", None)
        if maze_spec is None:
            return None
        return parse_maze_spec(maze_spec)
    finally:
        env.close()


def qpos_xy_to_maze_cell(xy: np.ndarray, maze_arr: np.ndarray | None = None) -> Tuple[int, int] | None:
    x = float(xy[0])
    y = float(xy[1])
    w = int(math.floor(x + 0.5))
    h = int(math.floor(y + 0.5))
    if maze_arr is not None:
        width, height = maze_arr.shape
        if w < 0 or w >= width or h < 0 or h >= height:
            return None
    return (w, h)


def point_in_wall_qpos_frame(maze_arr: np.ndarray, xy: np.ndarray) -> bool:
    x = float(xy[0])
    y = float(xy[1])
    wall_cells = np.argwhere(maze_arr == 10)
    for w, h in wall_cells:
        if (w - 0.5) <= x < (w + 0.5) and (h - 0.5) <= y < (h + 0.5):
            return True
    return False


def count_wall_hits_qpos_frame(maze_arr: np.ndarray | None, xy_seq: np.ndarray) -> int:
    if maze_arr is None:
        return 0
    xy_arr = np.asarray(xy_seq, dtype=np.float32)
    if xy_arr.ndim == 1:
        return int(point_in_wall_qpos_frame(maze_arr, xy_arr[:2]))
    return int(sum(point_in_wall_qpos_frame(maze_arr, xy_arr[i, :2]) for i in range(xy_arr.shape[0])))


def draw_maze_geometry(ax: plt.Axes, maze_arr: np.ndarray | None) -> None:
    if maze_arr is None:
        return
    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            is_wall = bool(maze_arr[w, h] == 10)
            face = "#686868" if is_wall else "#d9d9d9"
            edge = "#2f2f2f" if is_wall else "#bdbdbd"
            rect = mpatches.Rectangle(
                (w - 0.5, h - 0.5),
                1.0,
                1.0,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.35,
                alpha=0.9,
                zorder=0,
            )
            ax.add_patch(rect)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)


def collect_random_dataset(
    env_name: str,
    n_episodes: int,
    episode_len: int,
    action_scale: float,
    seed: int,
    corridor_aware_data: bool,
    corridor_max_resamples: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, float]]:
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)

    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = low * float(action_scale)
    action_high = high * float(action_scale)

    maze_arr = None
    can_rollback = False
    if corridor_aware_data:
        maze_spec = getattr(env.unwrapped, "str_maze_spec", None)
        if maze_spec is None:
            raise ValueError(
                "corridor-aware data requested, but environment has no str_maze_spec for wall parsing"
            )
        maze_arr = parse_maze_spec(maze_spec)
        can_rollback = hasattr(env.unwrapped, "set_state") and hasattr(env.unwrapped, "sim")

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminals: List[bool] = []
    timeouts: List[bool] = []
    episode_lengths: List[int] = []
    wall_rejects = 0
    failed_steps = 0

    for ep in range(n_episodes):
        obs = safe_reset(env, seed=seed + ep)
        ep_len = 0
        forced_episode_end = False
        for t in range(episode_len):
            if maze_arr is None:
                action = rng.uniform(action_low, action_high).astype(np.float32)
                next_obs, reward, done, _ = safe_step(env, action)
            else:
                qpos_before = env.unwrapped.sim.data.qpos.copy() if can_rollback else None
                qvel_before = env.unwrapped.sim.data.qvel.copy() if can_rollback else None

                accepted = False
                next_obs = None
                reward = 0.0
                done = False
                action = None
                for _ in range(corridor_max_resamples):
                    action_try = rng.uniform(action_low, action_high).astype(np.float32)
                    next_obs_try, reward_try, done_try, _ = safe_step(env, action_try)
                    if point_in_wall_qpos_frame(maze_arr, next_obs_try[:2]):
                        wall_rejects += 1
                        if can_rollback:
                            env.unwrapped.set_state(qpos_before, qvel_before)
                        continue
                    action = action_try
                    next_obs = next_obs_try
                    reward = reward_try
                    done = done_try
                    accepted = True
                    break

                if not accepted:
                    failed_steps += 1
                    forced_episode_end = True
                    break

            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(reward)
            terminals.append(done)

            is_timeout = (t == episode_len - 1) or done
            timeouts.append(is_timeout)

            obs = next_obs
            ep_len += 1
            if done:
                break

        # If corridor-aware rejection aborted the episode before writing a final step,
        # force a timeout boundary on the last recorded transition of this episode.
        if forced_episode_end and ep_len > 0:
            timeouts[-1] = True
            terminals[-1] = True

        episode_lengths.append(ep_len)

    env.close()

    dataset = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "timeouts": np.asarray(timeouts, dtype=np.bool_),
    }
    collection_stats = {
        "episode_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episode_len_min": int(np.min(episode_lengths)) if episode_lengths else 0,
        "episode_len_max": int(np.max(episode_lengths)) if episode_lengths else 0,
        "wall_rejects": int(wall_rejects),
        "failed_steps": int(failed_steps),
    }
    return dataset, action_low, action_high, collection_stats


def count_episodes_from_timeouts(timeouts: np.ndarray) -> int:
    return int(np.sum(np.asarray(timeouts, dtype=np.bool_)))


def build_goal_dataset_splits(
    raw_dataset: Dict[str, np.ndarray],
    cfg: Config,
    split_seed: int,
    device: torch.device,
) -> Tuple[GoalDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader, np.ndarray, np.ndarray]:
    n_episodes = max(1, count_episodes_from_timeouts(raw_dataset["timeouts"]))
    synthetic_env = SyntheticDatasetEnv(
        name=f"{cfg.env}-synthetic-replay",
        dataset=raw_dataset,
        max_episode_steps=max(cfg.episode_len, cfg.online_collect_episode_len),
    )
    dataset = GoalDataset(
        env=synthetic_env,
        horizon=cfg.horizon,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=max(cfg.max_path_length, cfg.online_collect_episode_len),
        max_n_episodes=n_episodes + 16,
        use_padding=False,
    )
    if len(dataset) == 0:
        raise ValueError(
            "GoalDataset produced zero samples. Increase episode length/max_path_length "
            "or reduce horizon when use_padding=False."
        )

    split_rng = np.random.default_rng(split_seed)
    all_idx = np.arange(len(dataset))
    split_rng.shuffle(all_idx)

    if len(all_idx) < 2:
        train_idx = all_idx.copy()
        val_idx = all_idx.copy()
    else:
        n_val = min(len(all_idx) - 1, max(1, int(len(all_idx) * cfg.val_frac)))
        val_idx = all_idx[:n_val]
        train_idx = all_idx[n_val:]

    train_subset = torch.utils.data.Subset(dataset, indices=train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, indices=val_idx.tolist())

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return dataset, train_loader, val_loader, train_idx, val_idx


def merge_replay_datasets(
    base: Dict[str, np.ndarray],
    additions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for key in ("observations", "actions", "rewards", "terminals", "timeouts"):
        if key not in base or key not in additions:
            raise KeyError(f"Missing replay key '{key}' in base/additions.")
        if base[key].size == 0:
            merged[key] = np.asarray(additions[key]).copy()
        elif additions[key].size == 0:
            merged[key] = np.asarray(base[key]).copy()
        else:
            merged[key] = np.concatenate([base[key], additions[key]], axis=0)
    return merged


def episode_spans_from_timeouts(timeouts: np.ndarray) -> List[Tuple[int, int]]:
    t = np.asarray(timeouts, dtype=np.bool_)
    spans: List[Tuple[int, int]] = []
    start = 0
    for i, timeout in enumerate(t):
        if bool(timeout):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(t):
        spans.append((start, len(t)))
    return spans


def sample_truncated_geometric_k(
    rng: np.random.Generator,
    p: float,
    min_k: int,
    max_k: int,
) -> int:
    if min_k <= 0:
        raise ValueError("min_k must be > 0")
    if max_k < min_k:
        raise ValueError("max_k must be >= min_k")
    if max_k == min_k:
        return int(min_k)
    for _ in range(128):
        k = int(rng.geometric(p))
        if min_k <= k <= max_k:
            return k
    return int(rng.integers(min_k, max_k + 1))


def sample_geometric_start_goal_pair(
    observations: np.ndarray,
    timeouts: np.ndarray,
    rng: np.random.Generator,
    geom_p: float,
    min_k: int,
    max_k: int,
    min_distance: float,
    max_attempts: int = 512,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    obs = np.asarray(observations, dtype=np.float32)
    spans = episode_spans_from_timeouts(timeouts)
    eligible_spans = [(a, b) for (a, b) in spans if (b - a) >= (min_k + 1)]
    if len(eligible_spans) == 0:
        raise ValueError("No eligible episode spans for geometric goal sampling.")

    for _ in range(max_attempts):
        s0, s1 = eligible_spans[int(rng.integers(0, len(eligible_spans)))]
        span_len = s1 - s0
        start_local = int(rng.integers(0, span_len - min_k))
        max_local_k = min(max_k, span_len - 1 - start_local)
        if max_local_k < min_k:
            continue
        k = sample_truncated_geometric_k(
            rng=rng,
            p=geom_p,
            min_k=min_k,
            max_k=max_local_k,
        )
        start_idx = s0 + start_local
        goal_idx = start_idx + k
        start_xy = obs[start_idx, :2].astype(np.float32)
        goal_xy = obs[goal_idx, :2].astype(np.float32)
        dist = float(np.linalg.norm(goal_xy - start_xy))
        if dist >= min_distance:
            return start_xy, goal_xy, int(k), dist

    # Fallback if geometric constraints are too strict in sparse replay.
    for _ in range(max_attempts):
        i0 = int(rng.integers(0, len(obs)))
        i1 = int(rng.integers(0, len(obs)))
        if i0 == i1:
            continue
        start_xy = obs[i0, :2].astype(np.float32)
        goal_xy = obs[i1, :2].astype(np.float32)
        dist = float(np.linalg.norm(goal_xy - start_xy))
        if dist >= min_distance:
            return start_xy, goal_xy, int(max(min_k, 1)), dist

    raise RuntimeError("Failed to sample start/goal pair from replay observations.")


def resolve_waypoint_t(planning_horizon: int, raw_waypoint_t: int) -> int:
    if int(planning_horizon) < 3:
        return max(0, int(planning_horizon) - 1)
    if int(raw_waypoint_t) > 0:
        t_wp = int(raw_waypoint_t)
    else:
        t_wp = int(planning_horizon) // 2
    return int(max(1, min(int(planning_horizon) - 2, t_wp)))


def sample_eval_waypoint(
    mode: str,
    replay_observations: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    waypoint_eps: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None

    pts = np.asarray(replay_observations, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    xy = pts[:, :2]
    start_xy = np.asarray(start_xy, dtype=np.float32).reshape(2)
    goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(2)

    min_sep = max(0.35, float(waypoint_eps) * 2.0)
    d_start = np.linalg.norm(xy - start_xy[None, :], axis=1)
    d_goal = np.linalg.norm(xy - goal_xy[None, :], axis=1)
    candidates = xy[(d_start >= min_sep) & (d_goal >= min_sep)]
    if len(candidates) == 0:
        candidates = xy
    feasible_wp = np.asarray(candidates[int(rng.integers(len(candidates)))], dtype=np.float32)
    if mode == "feasible":
        return feasible_wp
    if mode != "infeasible":
        raise ValueError(f"Unsupported eval waypoint mode: {mode}")

    # Best-effort infeasible waypoint: move far outside replay support box.
    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    span = np.maximum(maxs - mins, np.asarray([1e-3, 1e-3], dtype=np.float32))
    margin = np.maximum(0.5, span * 0.25)
    center = 0.5 * (mins + maxs)
    direction = feasible_wp - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.asarray([1.0, -1.0], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
    direction = direction / (norm + 1e-8)
    infeasible_wp = feasible_wp + direction * (span + margin)
    return np.asarray(infeasible_wp, dtype=np.float32)


@torch.no_grad()
def compute_val_loss(
    model: GaussianDiffusion,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_batches: int,
) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        batch = batch_to_device(batch, device=str(device))
        loss, _ = model.loss(*batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def normalize_condition(dataset: GoalDataset, obs: np.ndarray, device: torch.device) -> torch.Tensor:
    normed = dataset.normalizer.normalize(obs[None], "observations")
    return torch.as_tensor(normed, dtype=torch.float32, device=device)


@torch.no_grad()
def sample_imagined_trajectory(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    horizon: int,
    device: torch.device,
    n_samples: int,
    waypoint_xy: np.ndarray | None = None,
    waypoint_t: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    start_obs = np.asarray([start_xy[0], start_xy[1], 0.0, 0.0], dtype=np.float32)
    goal_obs = np.asarray([goal_xy[0], goal_xy[1], 0.0, 0.0], dtype=np.float32)

    cond = {
        0: normalize_condition(dataset, start_obs, device).repeat(n_samples, 1),
        horizon - 1: normalize_condition(dataset, goal_obs, device).repeat(n_samples, 1),
    }
    if waypoint_xy is not None:
        if waypoint_t is None:
            raise ValueError("waypoint_t must be provided when waypoint_xy is set")
        waypoint_obs = np.zeros(dataset.observation_dim, dtype=np.float32)
        waypoint_obs[:2] = np.asarray(waypoint_xy, dtype=np.float32)
        cond[int(waypoint_t)] = normalize_condition(dataset, waypoint_obs, device).repeat(n_samples, 1)

    model.eval()
    samples = model.conditional_sample(cond, horizon=horizon, verbose=False)
    if hasattr(samples, "trajectories"):
        samples = samples.trajectories
    elif isinstance(samples, (tuple, list)) and len(samples) > 0 and torch.is_tensor(samples[0]):
        samples = samples[0]
    if not torch.is_tensor(samples):
        raise TypeError(f"Unexpected conditional_sample output type: {type(samples)}")
    samples_np = samples.detach().cpu().numpy()

    actions_norm = samples_np[:, :, : dataset.action_dim]
    obs_norm = samples_np[:, :, dataset.action_dim :]
    actions = dataset.normalizer.unnormalize(actions_norm, "actions")
    observations = dataset.normalizer.unnormalize(obs_norm, "observations")
    return observations, actions


@torch.no_grad()
def sample_imagined_trajectory_from_obs(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    start_obs: np.ndarray,
    goal_xy: np.ndarray,
    horizon: int,
    device: torch.device,
    n_samples: int,
    waypoint_xy: np.ndarray | None = None,
    waypoint_t: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    start_obs = np.asarray(start_obs, dtype=np.float32).reshape(-1)
    if start_obs.shape[0] != dataset.observation_dim:
        raise ValueError(
            f"start_obs must have dim={dataset.observation_dim}, got shape={start_obs.shape}"
        )
    goal_obs = np.zeros(dataset.observation_dim, dtype=np.float32)
    goal_obs[:2] = np.asarray(goal_xy, dtype=np.float32)

    cond = {
        0: normalize_condition(dataset, start_obs, device).repeat(n_samples, 1),
        horizon - 1: normalize_condition(dataset, goal_obs, device).repeat(n_samples, 1),
    }
    if waypoint_xy is not None:
        if waypoint_t is None:
            raise ValueError("waypoint_t must be provided when waypoint_xy is set")
        waypoint_obs = np.zeros(dataset.observation_dim, dtype=np.float32)
        waypoint_obs[:2] = np.asarray(waypoint_xy, dtype=np.float32)
        cond[int(waypoint_t)] = normalize_condition(dataset, waypoint_obs, device).repeat(n_samples, 1)
    model.eval()
    samples = model.conditional_sample(cond, horizon=horizon, verbose=False)
    if hasattr(samples, "trajectories"):
        samples = samples.trajectories
    elif isinstance(samples, (tuple, list)) and len(samples) > 0 and torch.is_tensor(samples[0]):
        samples = samples[0]
    if not torch.is_tensor(samples):
        raise TypeError(f"Unexpected conditional_sample output type: {type(samples)}")
    samples_np = samples.detach().cpu().numpy()
    actions_norm = samples_np[:, :, : dataset.action_dim]
    obs_norm = samples_np[:, :, dataset.action_dim :]
    actions = dataset.normalizer.unnormalize(actions_norm, "actions")
    observations = dataset.normalizer.unnormalize(obs_norm, "observations")
    return observations, actions


@torch.no_grad()
def sample_best_plan_from_obs(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    start_obs: np.ndarray,
    goal_xy: np.ndarray,
    horizon: int,
    device: torch.device,
    maze_arr: np.ndarray | None,
    wall_aware_planning: bool,
    wall_aware_plan_samples: int,
    waypoint_xy: np.ndarray | None = None,
    waypoint_t: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n_candidates = 1
    if wall_aware_planning and maze_arr is not None:
        n_candidates = max(1, int(wall_aware_plan_samples))

    observations, actions = sample_imagined_trajectory_from_obs(
        model=model,
        dataset=dataset,
        start_obs=start_obs,
        goal_xy=goal_xy,
        horizon=horizon,
        device=device,
        n_samples=n_candidates,
        waypoint_xy=waypoint_xy,
        waypoint_t=waypoint_t,
    )

    best_idx = 0
    best_key = None
    for i in range(observations.shape[0]):
        xy = observations[i, :, :2]
        wall_hits = count_wall_hits_qpos_frame(maze_arr, xy)
        final_goal_err = float(np.linalg.norm(xy[-1] - goal_xy))
        key = (wall_hits, final_goal_err)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i

    selected_wall_hits = int(best_key[0]) if best_key is not None else 0
    return (
        np.asarray(observations[best_idx], dtype=np.float32),
        np.asarray(actions[best_idx], dtype=np.float32),
        selected_wall_hits,
    )


@torch.no_grad()
def collect_planner_dataset(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    env_name: str,
    replay_observations: np.ndarray,
    replay_timeouts: np.ndarray,
    horizon: int,
    device: torch.device,
    n_episodes: int,
    episode_len: int,
    transition_budget: int,
    replan_every_n_steps: int,
    goal_geom_p: float,
    goal_geom_min_k: int,
    goal_geom_max_k: int,
    goal_min_distance: float,
    seed: int,
    maze_arr: np.ndarray | None,
    wall_aware_planning: bool,
    wall_aware_plan_samples: int,
    planning_success_thresholds: Sequence[float],
    planning_success_rel_reduction: float,
    early_terminate_on_success: bool,
    early_terminate_threshold: float,
    min_accepted_episode_len: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    replan_stride = max(1, int(replan_every_n_steps))
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
    replans_per_episode: List[int] = []
    rollout_min_goal_dist: List[float] = []
    rollout_final_goal_dist: List[float] = []
    selected_plan_wall_hits: List[int] = []
    rollout_wall_hits: List[int] = []
    initial_goal_distances: List[float] = []
    final_distance_reduction_ratios: List[float] = []
    success_rel_reduction_flags: List[float] = []
    success_threshold_flags: Dict[float, List[float]] = {
        float(thr): [] for thr in planning_success_thresholds
    }

    # "episodes" and "transitions" refer to accepted data appended to replay.
    # We may attempt additional episodes that are rejected (e.g. too short) and are not
    # counted toward the env-step budget.
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

    # Safety against pathological rejection loops.
    # Allow ~10x attempts relative to the minimum number of required episodes.
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
                "This usually means early-termination + min-episode-len is rejecting too aggressively. "
                f"accepted_eps={accepted_episodes} accepted_transitions={accepted_transitions} "
                f"attempted_eps={attempted_episodes} rejected_short={rejected_short_episodes} "
                f"min_len={min_len} transition_budget={transition_budget}"
            )

        start_xy, goal_xy, goal_k, sampled_goal_dist = sample_geometric_start_goal_pair(
            observations=replay_observations,
            timeouts=replay_timeouts,
            rng=rng,
            geom_p=goal_geom_p,
            min_k=goal_geom_min_k,
            max_k=goal_geom_max_k,
            min_distance=goal_min_distance,
        )
        obs = reset_rollout_start(env, start_xy=start_xy)
        initial_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
        min_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
        final_goal_dist = min_goal_dist

        planned_actions = np.zeros((0, dataset.action_dim), dtype=np.float32)
        plan_offset = 0
        replan_count = 0
        ep_len = 0

        # Per-episode buffers so we can reject short episodes without counting them.
        ep_observations: List[np.ndarray] = []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []
        ep_terminals: List[bool] = []
        ep_timeouts: List[bool] = []
        ep_rollout_wall_hits: List[int] = []
        ep_dists_after_step: List[float] = []
        ep_replan_steps: List[int] = []
        ep_plan_wall_hits: List[int] = []

        for t in range(episode_len):
            should_replan = (t % replan_stride == 0) or (plan_offset >= len(planned_actions))
            if should_replan:
                _, best_actions, plan_wall_hits = sample_best_plan_from_obs(
                    model=model,
                    dataset=dataset,
                    start_obs=obs,
                    goal_xy=goal_xy,
                    horizon=horizon,
                    device=device,
                    maze_arr=maze_arr,
                    wall_aware_planning=wall_aware_planning,
                    wall_aware_plan_samples=wall_aware_plan_samples,
                )
                planned_actions = np.asarray(best_actions, dtype=np.float32)
                plan_offset = 0
                replan_count += 1
                ep_replan_steps.append(int(t))
                ep_plan_wall_hits.append(int(plan_wall_hits))

            if plan_offset >= len(planned_actions):
                action = rng.uniform(act_low, act_high).astype(np.float32)
            else:
                action = planned_actions[plan_offset].astype(np.float32)
                plan_offset += 1
            action = np.clip(action, act_low, act_high).astype(np.float32)

            next_obs, reward, done, _ = safe_step(env, action)
            attempted_transitions += 1
            dist = float(np.linalg.norm(next_obs[:2] - goal_xy))
            min_goal_dist = min(min_goal_dist, dist)
            final_goal_dist = dist
            wall_hit = int(count_wall_hits_qpos_frame(maze_arr, next_obs[:2]))

            ep_observations.append(obs.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(float(reward))
            ep_terminals.append(bool(done))
            ep_rollout_wall_hits.append(wall_hit)
            ep_dists_after_step.append(dist)

            hit_goal = bool(early_terminate_on_success and (dist <= float(early_terminate_threshold)))
            # Mark boundary if we ended because:
            # - max episode_len reached, or
            # - the env terminated (rare in Maze2D), or
            # - we reached the goal region and choose to terminate early (Option A).
            is_timeout = bool((t == episode_len - 1) or done or hit_goal)
            ep_timeouts.append(is_timeout)

            obs = next_obs
            ep_len += 1
            if done or hit_goal:
                break

        if ep_len > 0:
            ep_timeouts[-1] = True

        # Reject short episodes entirely (do not add to replay / do not count toward env budget).
        if ep_len < min_len:
            rejected_short_episodes += 1
            continue

        # If collecting by transition budget, optionally truncate the last accepted episode
        # so we do not exceed the budget. Avoid creating a too-short tail segment that would
        # be rejected anyway.
        if transition_budget > 0:
            remaining = int(transition_budget - accepted_transitions)
            if remaining <= 0:
                break
            if ep_len > remaining:
                if remaining < min_len:
                    # Not enough budget left to add a usable episode; stop.
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

                # Recompute distance stats on the accepted prefix.
                min_goal_dist = (
                    float(min(initial_goal_dist, float(np.min(ep_dists_after_step))))
                    if ep_dists_after_step
                    else float(initial_goal_dist)
                )
                final_goal_dist = float(ep_dists_after_step[-1]) if ep_dists_after_step else float("inf")

        # Accept episode: append to global replay buffers.
        observations.extend(ep_observations)
        actions.extend(ep_actions)
        rewards.extend(ep_rewards)
        terminals.extend(ep_terminals)
        timeouts.extend(ep_timeouts)
        rollout_wall_hits.extend(ep_rollout_wall_hits)
        # Keep only replans that occurred within the accepted prefix.
        for step_idx, wh in zip(ep_replan_steps, ep_plan_wall_hits):
            if step_idx < ep_len:
                selected_plan_wall_hits.append(int(wh))

        accepted_episodes += 1
        accepted_transitions += ep_len

        episode_lengths.append(ep_len)
        goal_distances.append(sampled_goal_dist)
        goal_ks.append(goal_k)
        replans_per_episode.append(int(sum(1 for s in ep_replan_steps if s < ep_len)))
        rollout_min_goal_dist.append(min_goal_dist)
        rollout_final_goal_dist.append(final_goal_dist)
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
    stats = {
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
        "replans_per_episode_mean": float(np.mean(replans_per_episode)) if replans_per_episode else 0.0,
        "selected_plan_wall_hits_mean": float(np.mean(selected_plan_wall_hits)) if selected_plan_wall_hits else float("nan"),
        "rollout_wall_hits_mean": float(np.mean(rollout_wall_hits)) if rollout_wall_hits else float("nan"),
        "initial_goal_distance_mean": float(np.mean(initial_goal_distances)) if initial_goal_distances else float("nan"),
        "rollout_min_goal_distance_mean": float(np.mean(rollout_min_goal_dist)) if rollout_min_goal_dist else float("nan"),
        "rollout_final_goal_distance_mean": float(np.mean(rollout_final_goal_dist)) if rollout_final_goal_dist else float("nan"),
        "planning_success_thresholds": ",".join([f"{float(thr):.4f}" for thr in planning_success_thresholds]),
        "planning_success_rel_reduction_target": float(planning_success_rel_reduction),
        "planning_final_distance_reduction_ratio_mean": float(np.mean(final_distance_reduction_ratios)) if final_distance_reduction_ratios else float("nan"),
        "planning_success_rate_final_rel_reduction": float(np.mean(success_rel_reduction_flags)) if success_rel_reduction_flags else float("nan"),
        f"planning_success_rate_final_rel{threshold_tag(planning_success_rel_reduction)}": (
            float(np.mean(success_rel_reduction_flags)) if success_rel_reduction_flags else float("nan")
        ),
    }
    for thr in planning_success_thresholds:
        tag = threshold_tag(float(thr))
        flags = success_threshold_flags[float(thr)]
        stats[f"planning_success_rate_final_t{tag}"] = float(np.mean(flags)) if flags else float("nan")
    return new_dataset, stats


def straightness_metrics(
    xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> Dict[str, float]:
    n = xy.shape[0]
    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
    line = start_xy[None, :] * (1.0 - alpha[:, None]) + goal_xy[None, :] * alpha[:, None]

    point_dev = np.linalg.norm(xy - line, axis=1)
    seg = np.diff(xy, axis=0)
    path_len = float(np.linalg.norm(seg, axis=1).sum()) if len(seg) > 0 else 0.0
    direct_len = float(np.linalg.norm(goal_xy - start_xy))
    straight_ratio = path_len / (direct_len + 1e-8)

    return {
        "mean_line_deviation": float(point_dev.mean()),
        "max_line_deviation": float(point_dev.max()),
        "final_goal_error": float(np.linalg.norm(xy[-1] - goal_xy)),
        "path_length": path_len,
        "direct_distance": direct_len,
        "path_over_direct": float(straight_ratio),
    }


def boundary_jump_ratios(xy: np.ndarray) -> Dict[str, float]:
    deltas = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if deltas.size == 0:
        return {"start_jump_ratio": float("nan"), "end_jump_ratio": float("nan")}
    if deltas.size <= 2:
        base = float(np.mean(deltas))
    else:
        base = float(np.mean(deltas[1:-1]))
    base = base + 1e-8
    return {
        "start_jump_ratio": float(deltas[0] / base),
        "end_jump_ratio": float(deltas[-1] / base),
    }


def transition_compatibility_metrics(
    observations: np.ndarray,
    actions: np.ndarray,
    goal_xy: np.ndarray,
    dt: float,
    goal_success_threshold: float,
) -> Dict[str, float]:
    xy = observations[:, :2]
    vel = observations[:, 2:4]

    if len(xy) >= 2:
        pregoal_xy = xy[-2]
    else:
        pregoal_xy = xy[-1]
    pregoal_error = float(np.linalg.norm(pregoal_xy - goal_xy))

    pos_delta = xy[1:] - xy[:-1]
    vel_pred = vel[:-1] * dt
    sv_err = np.linalg.norm(pos_delta - vel_pred, axis=1)
    sv_base = np.linalg.norm(pos_delta, axis=1) + 1e-8

    vel_delta = vel[1:] - vel[:-1]
    acc_pred = actions[:-1] * dt
    va_err = np.linalg.norm(vel_delta - acc_pred, axis=1)
    va_base = np.linalg.norm(vel_delta, axis=1) + 1e-8

    return {
        "pregoal_error": pregoal_error,
        "pregoal_success": float(pregoal_error <= goal_success_threshold),
        "state_velocity_l2_mean": float(np.mean(sv_err)) if sv_err.size else float("nan"),
        "state_velocity_rel_mean": float(np.mean(sv_err / sv_base)) if sv_err.size else float("nan"),
        "velocity_action_l2_mean": float(np.mean(va_err)) if va_err.size else float("nan"),
        "velocity_action_rel_mean": float(np.mean(va_err / va_base)) if va_err.size else float("nan"),
    }


def reset_rollout_start(env: gym.Env, start_xy: np.ndarray) -> np.ndarray:
    # Keep gym wrappers (e.g., OrderEnforcing) in a valid stepped-after-reset state.
    _ = safe_reset(env)
    base_env = env.unwrapped
    if hasattr(base_env, "reset_to_location"):
        obs = base_env.reset_to_location(np.asarray(start_xy, dtype=np.float32))
        return np.asarray(obs, dtype=np.float32)
    return safe_reset(env)


@torch.no_grad()
def rollout_to_goal(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    rollout_env: gym.Env,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    planning_horizon: int,
    rollout_horizon: int,
    device: torch.device,
    rollout_mode: str,
    rollout_replan_every_n_steps: int,
    maze_arr: np.ndarray | None,
    wall_aware_planning: bool,
    wall_aware_plan_samples: int,
    open_loop_actions: np.ndarray | None = None,
    waypoint_xy: np.ndarray | None = None,
    waypoint_t: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    obs = reset_rollout_start(rollout_env, start_xy=start_xy)
    min_goal_dist = float(np.linalg.norm(obs[:2] - goal_xy))
    final_goal_dist = min_goal_dist
    rollout_wall_hits = count_wall_hits_qpos_frame(maze_arr, obs[:2])

    act_low = np.asarray(rollout_env.action_space.low, dtype=np.float32)
    act_high = np.asarray(rollout_env.action_space.high, dtype=np.float32)
    replan_stride = max(1, int(rollout_replan_every_n_steps))

    rollout_xy: List[np.ndarray] = [obs[:2].copy()]
    rollout_actions: List[np.ndarray] = []

    if rollout_mode == "open_loop":
        if open_loop_actions is None:
            raise ValueError("open_loop rollout requires open_loop_actions.")
        open_loop_actions = np.asarray(open_loop_actions, dtype=np.float32)
        for t in range(rollout_horizon):
            if t < len(open_loop_actions):
                act = open_loop_actions[t]
            else:
                act = np.zeros(dataset.action_dim, dtype=np.float32)
            action = np.clip(act, act_low, act_high).astype(np.float32)
            obs, _, _, _ = safe_step(rollout_env, action)
            dist = float(np.linalg.norm(obs[:2] - goal_xy))
            min_goal_dist = min(min_goal_dist, dist)
            final_goal_dist = dist
            rollout_wall_hits += count_wall_hits_qpos_frame(maze_arr, obs[:2])
            rollout_actions.append(action.copy())
            rollout_xy.append(obs[:2].copy())
    elif rollout_mode == "receding_horizon":
        planned_actions = np.zeros((0, dataset.action_dim), dtype=np.float32)
        plan_offset = 0
        for t in range(rollout_horizon):
            should_replan = (t % replan_stride == 0) or (plan_offset >= len(planned_actions))
            if should_replan:
                _, best_actions, _ = sample_best_plan_from_obs(
                    model=model,
                    dataset=dataset,
                    start_obs=obs,
                    goal_xy=goal_xy,
                    horizon=planning_horizon,
                    device=device,
                    maze_arr=maze_arr,
                    wall_aware_planning=wall_aware_planning,
                    wall_aware_plan_samples=wall_aware_plan_samples,
                    waypoint_xy=waypoint_xy,
                    waypoint_t=waypoint_t,
                )
                planned_actions = np.asarray(best_actions, dtype=np.float32)
                plan_offset = 0

            if plan_offset >= len(planned_actions):
                action = np.zeros(dataset.action_dim, dtype=np.float32)
            else:
                action = planned_actions[plan_offset].astype(np.float32)
                plan_offset += 1
            action = np.clip(action, act_low, act_high).astype(np.float32)

            obs, _, _, _ = safe_step(rollout_env, action)
            dist = float(np.linalg.norm(obs[:2] - goal_xy))
            min_goal_dist = min(min_goal_dist, dist)
            final_goal_dist = dist
            rollout_wall_hits += count_wall_hits_qpos_frame(maze_arr, obs[:2])
            rollout_actions.append(action.copy())
            rollout_xy.append(obs[:2].copy())
    else:
        raise ValueError(f"Unsupported rollout_mode='{rollout_mode}'")

    return (
        np.asarray(rollout_xy, dtype=np.float32),
        np.asarray(rollout_actions, dtype=np.float32),
        min_goal_dist,
        final_goal_dist,
        int(rollout_wall_hits),
    )


def rollout_prefix_distance_stats(
    rollout_xy: np.ndarray,
    goal_xy: np.ndarray,
    prefix_horizons: Sequence[int],
) -> Dict[int, Tuple[float, float]]:
    xy = np.asarray(rollout_xy, dtype=np.float32)[:, :2]
    dists = np.linalg.norm(xy - np.asarray(goal_xy, dtype=np.float32)[None, :], axis=1)
    stats: Dict[int, Tuple[float, float]] = {}
    for h in prefix_horizons:
        idx = min(int(h), len(dists) - 1)
        stats[int(h)] = (float(np.min(dists[: idx + 1])), float(dists[idx]))
    return stats


@torch.no_grad()
def evaluate_goal_progress(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    env_name: str,
    replay_observations: np.ndarray,
    query_pairs: List[Tuple[np.ndarray, np.ndarray]],
    planning_horizon: int,
    rollout_horizon: int,
    success_prefix_horizons: Sequence[int],
    device: torch.device,
    n_samples: int,
    goal_success_threshold: float,
    rollout_mode: str,
    rollout_replan_every_n_steps: int,
    maze_arr: np.ndarray | None,
    wall_aware_planning: bool,
    wall_aware_plan_samples: int,
    eval_waypoint_mode: str = "none",
    eval_waypoint_t: int = 0,
    eval_waypoint_eps: float = 0.2,
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
    # Goal-coverage signals:
    # - query coverage: fraction of queried start-goal pairs with >=1 successful sample.
    # - cell coverage: fraction of unique queried goal cells reached by >=1 sample.
    query_success_any_by_prefix: Dict[int, np.ndarray] = {
        int(h): np.zeros(len(query_pairs), dtype=np.bool_) for h in success_prefix_horizons
    }
    query_goal_cells: List[Tuple[int, int] | None] = [
        qpos_xy_to_maze_cell(goal_xy, maze_arr=maze_arr) for _, goal_xy in query_pairs
    ]
    imagined_wall_hits: List[int] = []
    rollout_wall_hits: List[int] = []
    waypoint_hits: List[float] = []
    waypoint_min_distances: List[float] = []
    waypoint_t_effective = resolve_waypoint_t(int(planning_horizon), int(eval_waypoint_t))
    waypoint_rng = np.random.default_rng(20260219 + int(planning_horizon) + len(query_pairs))

    rollout_env = gym.make(env_name)
    dt = float(getattr(rollout_env.unwrapped, "dt", 1.0))

    for qid, (start_xy, goal_xy) in enumerate(query_pairs):
        waypoint_xy = sample_eval_waypoint(
            mode=eval_waypoint_mode,
            replay_observations=replay_observations,
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoint_eps=float(eval_waypoint_eps),
            rng=waypoint_rng,
        )
        waypoint_t = waypoint_t_effective if waypoint_xy is not None else None
        observations, actions = sample_imagined_trajectory(
            model=model,
            dataset=dataset,
            start_xy=start_xy,
            goal_xy=goal_xy,
            horizon=planning_horizon,
            device=device,
            n_samples=n_samples,
            waypoint_xy=waypoint_xy,
            waypoint_t=waypoint_t,
        )

        for sid in range(observations.shape[0]):
            obs_traj = observations[sid]
            act_traj = actions[sid]
            xy = obs_traj[:, :2]
            s = straightness_metrics(xy=xy, start_xy=start_xy, goal_xy=goal_xy)
            b = boundary_jump_ratios(xy)
            c = transition_compatibility_metrics(
                observations=obs_traj,
                actions=act_traj,
                goal_xy=goal_xy,
                dt=dt,
                goal_success_threshold=goal_success_threshold,
            )

            imagined_goal_error = s["final_goal_error"]
            imagined_successes.append(float(imagined_goal_error <= goal_success_threshold))
            imagined_goal_errors.append(imagined_goal_error)
            imagined_wall_hits.append(count_wall_hits_qpos_frame(maze_arr, obs_traj[:, :2]))
            imagined_pregoal_successes.append(c["pregoal_success"])
            imagined_pregoal_errors.append(c["pregoal_error"])
            imagined_line_dev.append(s["mean_line_deviation"])
            imagined_path_ratio.append(s["path_over_direct"])
            start_jump_ratios.append(b["start_jump_ratio"])
            end_jump_ratios.append(b["end_jump_ratio"])
            state_velocity_l2_means.append(c["state_velocity_l2_mean"])
            state_velocity_rel_means.append(c["state_velocity_rel_mean"])
            velocity_action_l2_means.append(c["velocity_action_l2_mean"])
            velocity_action_rel_means.append(c["velocity_action_rel_mean"])

            rollout_xy, rollout_actions, min_goal_dist, final_goal_dist, rollout_wall_hit_count = rollout_to_goal(
                model=model,
                dataset=dataset,
                rollout_env=rollout_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                planning_horizon=planning_horizon,
                rollout_horizon=rollout_horizon,
                device=device,
                rollout_mode=rollout_mode,
                rollout_replan_every_n_steps=rollout_replan_every_n_steps,
                maze_arr=maze_arr,
                wall_aware_planning=wall_aware_planning,
                wall_aware_plan_samples=wall_aware_plan_samples,
                open_loop_actions=act_traj if rollout_mode == "open_loop" else None,
                waypoint_xy=waypoint_xy,
                waypoint_t=waypoint_t,
            )
            if waypoint_xy is not None:
                waypoint_d = np.linalg.norm(
                    np.asarray(rollout_xy, dtype=np.float32)[:, :2] - np.asarray(waypoint_xy, dtype=np.float32)[None, :],
                    axis=1,
                )
                waypoint_min = float(np.min(waypoint_d))
                waypoint_min_distances.append(waypoint_min)
                waypoint_hits.append(float(waypoint_min <= float(eval_waypoint_eps)))

            # Essential protocol choice: compute all success@h metrics from the same
            # realized rollout so @64/@128/@192/@256 are directly comparable.
            prefix_stats = rollout_prefix_distance_stats(
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
            del rollout_xy, rollout_actions

            rollout_successes.append(float(min_goal_dist <= goal_success_threshold))
            rollout_min_goal_distances.append(min_goal_dist)
            rollout_final_goal_errors.append(final_goal_dist)
            rollout_wall_hits.append(int(rollout_wall_hit_count))

    rollout_env.close()

    eval_num_queries = int(len(query_pairs))
    eval_samples_per_query = int(n_samples)
    eval_num_trajectories = int(eval_num_queries * eval_samples_per_query)
    rollout_success_count = int(np.sum(rollout_successes)) if rollout_successes else 0
    goal_cell_total = int(len({cell for cell in query_goal_cells if cell is not None}))

    metrics = {
        "eval_num_queries": eval_num_queries,
        "eval_samples_per_query": eval_samples_per_query,
        "eval_num_trajectories": eval_num_trajectories,
        "eval_unique_goal_cells": goal_cell_total,
        "eval_rollout_mode": rollout_mode,
        "eval_rollout_replan_every_n_steps": int(rollout_replan_every_n_steps),
        "eval_rollout_horizon": int(rollout_horizon),
        "eval_success_prefix_horizons": ",".join(str(int(h)) for h in success_prefix_horizons),
        "eval_waypoint_mode": str(eval_waypoint_mode),
        "eval_waypoint_t": int(waypoint_t_effective),
        "eval_waypoint_eps": float(eval_waypoint_eps),
        "waypoint_hit_rate": float(np.mean(waypoint_hits)) if waypoint_hits else float("nan"),
        "waypoint_min_distance_mean": float(np.mean(waypoint_min_distances)) if waypoint_min_distances else float("nan"),
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


def plot_losses(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["step"], metrics_df["train_loss"], label="train_loss", alpha=0.8)
    val_df = metrics_df[np.isfinite(metrics_df["val_loss"])]
    if len(val_df) > 0:
        ax.plot(val_df["step"], val_df["val_loss"], marker="o", linewidth=1.5, label="val_loss")
    ax.set_xlabel("train step")
    ax.set_ylabel("loss")
    ax.set_title("Synthetic Maze2D: train/val loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_progress(progress_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(progress_df["step"], progress_df["imagined_goal_success_rate"], marker="o", label="imagined")
    if "imagined_pregoal_success_rate" in progress_df:
        axes[0].plot(progress_df["step"], progress_df["imagined_pregoal_success_rate"], marker="o", label="imagined-pregoal")
    rollout_prefix_cols = sorted(
        [c for c in progress_df.columns if c.startswith("rollout_goal_success_rate_h")],
        key=lambda c: int(c.rsplit("h", 1)[-1]),
    )
    if rollout_prefix_cols:
        for c in rollout_prefix_cols:
            h = int(c.rsplit("h", 1)[-1])
            axes[0].plot(progress_df["step"], progress_df[c], marker="o", label=f"rollout@{h}")
    else:
        axes[0].plot(progress_df["step"], progress_df["rollout_goal_success_rate"], marker="o", label="rollout")
    axes[0].set_title("Goal Success Rate")
    axes[0].set_xlabel("train step")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(progress_df["step"], progress_df["imagined_goal_error_mean"], marker="o", label="imagined")
    if "imagined_pregoal_error_mean" in progress_df:
        axes[1].plot(progress_df["step"], progress_df["imagined_pregoal_error_mean"], marker="o", label="imagined-pregoal")
    axes[1].plot(progress_df["step"], progress_df["rollout_final_goal_error_mean"], marker="o", label="rollout-final")
    axes[1].plot(progress_df["step"], progress_df["rollout_min_goal_distance_mean"], marker="o", label="rollout-min")
    axes[1].set_title("Goal Distance Metrics")
    axes[1].set_xlabel("train step")
    axes[1].set_ylabel("distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if "state_velocity_rel_mean" in progress_df and "velocity_action_rel_mean" in progress_df:
        axes[2].plot(progress_df["step"], progress_df["state_velocity_rel_mean"], marker="o", label="state-velocity rel")
        axes[2].plot(progress_df["step"], progress_df["velocity_action_rel_mean"], marker="o", label="velocity-action rel")
        axes[2].set_title("Transition Compatibility (Relative Error)")
        axes[2].set_xlabel("train step")
        axes[2].set_ylabel("relative error")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
    else:
        axes[2].plot(progress_df["step"], progress_df["boundary_start_jump_ratio_mean"], marker="o", label="start-jump")
        axes[2].plot(progress_df["step"], progress_df["boundary_end_jump_ratio_mean"], marker="o", label="end-jump")
        axes[2].set_title("Boundary Compatibility (Jump Ratios)")
        axes[2].set_xlabel("train step")
        axes[2].set_ylabel("ratio")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_query_trajectories(
    query_rows: List[Dict[str, float]],
    out_path: Path,
    maze_arr: np.ndarray | None = None,
) -> None:
    n = len(query_rows)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows), squeeze=False)

    for i, row in enumerate(query_rows):
        ax = axes[i // cols][i % cols]
        draw_maze_geometry(ax, maze_arr=maze_arr)
        xy = np.asarray(json.loads(row["xy_json"]), dtype=np.float32)
        start_xy = np.asarray([row["start_x"], row["start_y"]], dtype=np.float32)
        goal_xy = np.asarray([row["goal_x"], row["goal_y"]], dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, len(xy), dtype=np.float32)
        line = start_xy[None, :] * (1.0 - alpha[:, None]) + goal_xy[None, :] * alpha[:, None]

        ax.plot(xy[:, 0], xy[:, 1], linewidth=2.0, color="#1f77b4", label="imagined")
        if "rollout_xy_json" in row and isinstance(row["rollout_xy_json"], str) and len(row["rollout_xy_json"]) > 2:
            rollout_xy = np.asarray(json.loads(row["rollout_xy_json"]), dtype=np.float32)
            ax.plot(rollout_xy[:, 0], rollout_xy[:, 1], linewidth=2.0, color="#ff7f0e", alpha=0.9, label="rollout")
        ax.plot(line[:, 0], line[:, 1], "--", linewidth=1.4, color="black", alpha=0.8, label="straight line")
        ax.scatter([start_xy[0]], [start_xy[1]], s=70, color="green", marker="o", edgecolors="black", linewidths=0.4)
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=85, color="red", marker="X", edgecolors="black", linewidths=0.4)
        ax.scatter(
            xy[[0, len(xy) // 2, -1], 0],
            xy[[0, len(xy) // 2, -1], 1],
            s=[36, 30, 42],
            color=["#1f77b4", "#4169e1", "#1f77b4"],
            alpha=0.9,
        )
        imag_wall = int(float(row.get("imagined_in_wall_points", 0.0)))
        roll_wall = int(float(row.get("rollout_in_wall_points", 0.0)))
        ax.set_title(
            f"q{int(row['query_id'])} "
            f"dev={row['mean_line_deviation']:.3f} "
            f"ratio={row['path_over_direct']:.3f} "
            f"rmin={row.get('rollout_min_goal_distance', float('nan')):.3f} "
            f"iw={imag_wall} rw={roll_wall}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    replay_import_path = str(cfg.replay_import_path or cfg.replay_load_npz).strip()
    replay_export_path = str(cfg.replay_export_path or cfg.replay_save_npz).strip()
    if cfg.replay_import_path and cfg.replay_load_npz and str(cfg.replay_import_path) != str(cfg.replay_load_npz):
        raise ValueError(
            f"Conflicting replay import flags: --replay_import_path={cfg.replay_import_path} "
            f"vs --replay_load_npz={cfg.replay_load_npz}"
        )
    if cfg.replay_export_path and cfg.replay_save_npz and str(cfg.replay_export_path) != str(cfg.replay_save_npz):
        raise ValueError(
            f"Conflicting replay export flags: --replay_export_path={cfg.replay_export_path} "
            f"vs --replay_save_npz={cfg.replay_save_npz}"
        )
    # Ensure prints show up promptly even when stdout/stderr are piped (e.g. via `tee`).
    try:
        import sys

        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
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
    if cfg.wall_aware_plan_samples <= 0:
        raise ValueError("--wall_aware_plan_samples must be > 0")
    if cfg.eval_rollout_horizon <= 0:
        raise ValueError("--eval_rollout_horizon must be > 0")
    if cfg.online_collect_transition_budget_per_round < 0:
        raise ValueError("--online_collect_transition_budget_per_round must be >= 0")
    if cfg.online_early_terminate_threshold <= 0.0:
        raise ValueError("--online_early_terminate_threshold must be > 0")
    if cfg.online_min_accepted_episode_len < 0:
        raise ValueError("--online_min_accepted_episode_len must be >= 0")
    if not (0.0 <= cfg.online_planning_success_rel_reduction <= 1.0):
        raise ValueError("--online_planning_success_rel_reduction must be in [0, 1]")
    if cfg.eval_waypoint_eps <= 0.0:
        raise ValueError("--eval_waypoint_eps must be > 0")
    if cfg.eval_waypoint_mode not in {"none", "feasible", "infeasible"}:
        raise ValueError("--eval_waypoint_mode must be one of: none, feasible, infeasible")

    online_min_accepted_episode_len = int(cfg.online_min_accepted_episode_len) if int(cfg.online_min_accepted_episode_len) > 0 else int(cfg.horizon)
    if online_min_accepted_episode_len <= 0:
        raise ValueError("online_min_accepted_episode_len must be > 0")
    if online_min_accepted_episode_len > int(cfg.online_collect_episode_len):
        raise ValueError(
            "--online_min_accepted_episode_len must be <= --online_collect_episode_len "
            f"(min_accepted={online_min_accepted_episode_len}, collect_episode_len={cfg.online_collect_episode_len})"
        )
    if cfg.online_collect_transition_budget_per_round > 0 and cfg.online_collect_transition_budget_per_round < online_min_accepted_episode_len:
        raise ValueError(
            "--online_collect_transition_budget_per_round must be >= --online_min_accepted_episode_len "
            f"(budget={cfg.online_collect_transition_budget_per_round}, min_accepted={online_min_accepted_episode_len})"
        )

    eval_success_prefix_horizons = tuple(
        int(h) for h in parse_int_list(cfg.eval_success_prefix_horizons, "--eval_success_prefix_horizons")
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
        for thr in parse_float_list(cfg.online_planning_success_thresholds, "--online_planning_success_thresholds")
    )
    if any(thr <= 0.0 for thr in online_planning_success_thresholds):
        raise ValueError("--online_planning_success_thresholds must be > 0")
    if cfg.fixed_replay_snapshot_round < 0:
        raise ValueError("--fixed_replay_snapshot_round must be >= 0")
    if replay_import_path and not Path(replay_import_path).is_file():
        raise FileNotFoundError(
            f"Replay import file not found: {replay_import_path} "
            f"(from --replay_import_path/--replay_load_npz)"
        )
    if cfg.collector_ckpt_path and not Path(cfg.collector_ckpt_path).is_file():
        raise FileNotFoundError(f"--collector_ckpt_path not found: {cfg.collector_ckpt_path}")

    logdir = make_logdir(cfg)
    with open(logdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] env={cfg.env} seed={cfg.seed}")
    print(f"[setup] device={device}")
    print(f"[setup] logdir={logdir}")
    print(
        "[setup] "
        f"collector_weights={cfg.collector_weights} "
        f"eval_weights={cfg.eval_weights} "
        f"collector_ckpt_path={cfg.collector_ckpt_path or 'none'} "
        f"collector_ckpt_weights={cfg.collector_ckpt_weights}"
    )
    print(
        "[setup] "
        f"replay_import_path={replay_import_path or 'none'} "
        f"replay_export_path={replay_export_path or 'none'} "
        f"fixed_replay_snapshot_round={cfg.fixed_replay_snapshot_round} "
        f"fixed_replay_snapshot_npz={cfg.fixed_replay_snapshot_npz or 'default'} "
        f"disable_online_collection={int(bool(cfg.disable_online_collection))}"
    )
    print(
        "[setup] "
        f"eval_waypoint_mode={cfg.eval_waypoint_mode} "
        f"eval_waypoint_t={cfg.eval_waypoint_t} "
        f"eval_waypoint_eps={cfg.eval_waypoint_eps:.3f}"
    )
    maze_arr = load_maze_arr_from_env(cfg.env)
    if maze_arr is None:
        print("[setup] maze geometry unavailable from environment; wall-aware planning disabled.")
    else:
        print(
            f"[setup] maze geometry loaded: shape={maze_arr.shape} "
            f"wall_cells={int(np.sum(maze_arr == 10))}"
        )

    replay_import_meta: Dict[str, Any] = {}
    if replay_import_path:
        raw_dataset, action_low, action_high, collection_stats, replay_import_meta = load_replay_artifact(
            Path(replay_import_path)
        )
        print(f"[data] loaded replay_artifact={replay_import_path}")
        print(
            "[replay] imported "
            f"transitions={int(replay_import_meta.get('transitions', len(raw_dataset['observations'])))} "
            f"episodes={int(replay_import_meta.get('episodes', count_episodes_from_timeouts(raw_dataset['timeouts'])))} "
            f"fingerprint={replay_import_meta.get('fingerprint', 'na')}"
        )
    else:
        raw_dataset, action_low, action_high, collection_stats = collect_random_dataset(
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
        f"episodes={count_episodes_from_timeouts(raw_dataset['timeouts'])}"
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

    if replay_import_path:
        # With use_padding=False, GoalDataset needs at least one episode with length > horizon,
        # otherwise it yields zero windows and downstream training silently becomes nonsense.
        ep_max = int(collection_stats.get("episode_len_max", 0) or 0)
        if ep_max > 0 and int(cfg.horizon) >= ep_max:
            raise ValueError(
                f"Replay import {replay_import_path} has episode_len_max={ep_max}, but --horizon={cfg.horizon}. "
                "With use_padding=False this produces zero training windows. "
                "Set --horizon < episode_len_max (or load a replay with longer episodes)."
            )

    dataset, train_loader, val_loader, train_idx, val_idx = build_goal_dataset_splits(
        raw_dataset=raw_dataset,
        cfg=cfg,
        split_seed=cfg.seed + 1337,
        device=device,
    )
    initial_train_samples = int(len(train_idx))
    initial_val_samples = int(len(val_idx))
    print(f"[data] samples total={len(dataset)} train={initial_train_samples} val={initial_val_samples}")

    dim_mults = parse_dim_mults(cfg.model_dim_mults)
    model = TemporalUnet(
        horizon=cfg.horizon,
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=dataset.observation_dim,
        dim=cfg.model_dim,
        dim_mults=dim_mults,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=cfg.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=cfg.n_diffusion_steps,
        clip_denoised=cfg.clip_denoised,
        predict_epsilon=cfg.predict_epsilon,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
    ).to(device)

    ema_helper = EMA(cfg.ema_decay)
    ema_model = copy.deepcopy(diffusion).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.learning_rate)

    collector_ckpt_model: GaussianDiffusion | None = None
    if cfg.collector_ckpt_path:
        ckpt = torch.load(cfg.collector_ckpt_path, map_location=device)
        ckpt_key = "ema" if cfg.collector_ckpt_weights == "ema" else "model"
        if ckpt_key not in ckpt:
            raise KeyError(
                f"Collector checkpoint missing key '{ckpt_key}': {cfg.collector_ckpt_path} "
                f"(keys={sorted(list(ckpt.keys()))})"
            )
        collector_ckpt_model = copy.deepcopy(diffusion).to(device)
        collector_ckpt_model.load_state_dict(ckpt[ckpt_key])
        collector_ckpt_model.eval()
        for p in collector_ckpt_model.parameters():
            p.requires_grad_(False)
        print(
            "[collector] "
            f"mode=ckpt path={cfg.collector_ckpt_path} "
            f"weights={cfg.collector_ckpt_weights} "
            f"(ckpt_key={ckpt_key})"
        )

    def model_for_eval() -> GaussianDiffusion:
        return ema_model if cfg.eval_weights == "ema" else diffusion

    def model_for_collection() -> GaussianDiffusion:
        if collector_ckpt_model is not None:
            return collector_ckpt_model
        return ema_model if cfg.collector_weights == "ema" else diffusion

    def replay_metadata(stage: str, round_idx: int | None = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "env_id": str(cfg.env),
            "seed": int(cfg.seed),
            "collector_method": "diffuser",
            "collector_weights": str(cfg.collector_weights),
            "collection_budget": {
                "offline_n_episodes": int(cfg.n_episodes),
                "offline_episode_len": int(cfg.episode_len),
                "online_rounds": int(cfg.online_rounds),
                "online_collect_episodes_per_round": int(cfg.online_collect_episodes_per_round),
                "online_collect_transition_budget_per_round": int(cfg.online_collect_transition_budget_per_round),
                "online_collect_episode_len": int(cfg.online_collect_episode_len),
            },
            "normalization": "raw_env_space; training normalizer=LimitsNormalizer",
            "stage": str(stage),
            "source_replay_fingerprint": replay_import_meta.get("fingerprint", ""),
        }
        if round_idx is not None:
            meta["round_idx"] = int(round_idx)
        return meta

    metrics_rows: List[Dict[str, float]] = []
    online_collection_rows: List[Dict[str, float]] = []
    if cfg.query_mode == "fixed":
        query_bank: List[Tuple[np.ndarray, np.ndarray]] = parse_queries(cfg.query)
        print(f"[eval-query] mode=fixed num_pairs={len(query_bank)}")
    else:
        bank_size = max(cfg.query_bank_size, cfg.num_eval_queries)
        query_bank = build_diverse_query_bank(
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
        return select_query_pairs(
            query_bank=query_bank,
            num_queries=cfg.num_eval_queries,
            seed=q_seed,
        )

    global_step = 0
    last_eval_query_pairs = query_pairs_for_step(step=0)
    progress_rows: List[Dict[str, float]] = []
    current_dataset = dataset

    def run_training_steps(
        num_steps: int,
        train_loader_cur: torch.utils.data.DataLoader,
        val_loader_cur: torch.utils.data.DataLoader,
        dataset_for_eval: GoalDataset,
        phase: str,
    ) -> None:
        nonlocal global_step, last_eval_query_pairs
        train_iter = cycle(train_loader_cur)
        for _ in range(num_steps):
            global_step += 1
            diffusion.train()
            optimizer.zero_grad(set_to_none=True)
            batch = next(train_iter)
            batch = batch_to_device(batch, device=str(device))
            loss, infos = diffusion.loss(*batch)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
            optimizer.step()

            if global_step % cfg.ema_update_every == 0:
                if global_step < cfg.ema_start_step:
                    ema_model.load_state_dict(diffusion.state_dict())
                else:
                    ema_helper.update_model_average(ema_model, diffusion)

            val_loss = float("nan")
            if global_step == 1 or (cfg.val_every > 0 and global_step % cfg.val_every == 0):
                val_loss = compute_val_loss(
                    model=model_for_eval(),
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
            for k, v in infos.items():
                row[f"info_{k}"] = float(v)
            metrics_rows.append(row)

            if global_step == 1 or global_step % 100 == 0 or np.isfinite(val_loss):
                print(
                    f"[train] phase={phase} step={global_step:5d} "
                    f"train_loss={row['train_loss']:.5f} "
                    f"val_loss={row['val_loss']:.5f}"
                )

            if cfg.save_checkpoint_every > 0 and global_step % cfg.save_checkpoint_every == 0:
                checkpoint_step = {
                    "step": int(global_step),
                    "phase": phase,
                    "model": diffusion.state_dict(),
                    "ema": ema_model.state_dict(),
                    "config": asdict(cfg),
                }
                torch.save(checkpoint_step, logdir / f"checkpoint_step{global_step}.pt")

            if cfg.eval_goal_every > 0 and global_step % cfg.eval_goal_every == 0:
                eval_query_pairs = query_pairs_for_step(step=global_step)
                last_eval_query_pairs = eval_query_pairs
                # Essential protocol choice: evaluate all prefix horizons from one
                # shared rollout realization to avoid stochastic confounds across
                # separate reruns at different horizons.
                progress = evaluate_goal_progress(
                    model=model_for_eval(),
                    dataset=dataset_for_eval,
                    env_name=cfg.env,
                    replay_observations=raw_dataset["observations"],
                    query_pairs=eval_query_pairs,
                    planning_horizon=cfg.horizon,
                    rollout_horizon=cfg.eval_rollout_horizon,
                    success_prefix_horizons=eval_success_prefix_horizons,
                    device=device,
                    n_samples=cfg.query_batch_size,
                    goal_success_threshold=cfg.goal_success_threshold,
                    rollout_mode=cfg.eval_rollout_mode,
                    rollout_replan_every_n_steps=cfg.eval_rollout_replan_every_n_steps,
                    maze_arr=maze_arr,
                    wall_aware_planning=cfg.wall_aware_planning,
                    wall_aware_plan_samples=cfg.wall_aware_plan_samples,
                    eval_waypoint_mode=cfg.eval_waypoint_mode,
                    eval_waypoint_t=cfg.eval_waypoint_t,
                    eval_waypoint_eps=cfg.eval_waypoint_eps,
                )
                progress["step"] = int(global_step)
                progress["phase"] = phase
                progress["eval_query_mode"] = cfg.query_mode
                progress["eval_query_pairs"] = int(len(eval_query_pairs))
                progress_rows.append(progress)
                # Persist progress incrementally so monitor/auto-decider loops can
                # observe intermediate results (otherwise this file only exists
                # after the entire run finishes, which defeats online monitoring).
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
                    f"vel_act_rel={progress['velocity_action_rel_mean']:.3f} "
                    f"waypoint_hit={progress.get('waypoint_hit_rate', float('nan')):.3f}"
                )

    run_training_steps(
        num_steps=cfg.train_steps,
        train_loader_cur=train_loader,
        val_loader_cur=val_loader,
        dataset_for_eval=current_dataset,
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
            f"replan_every_n_steps={cfg.online_replan_every_n_steps} "
            f"train_steps_per_round={cfg.online_train_steps_per_round}"
        )
        replay_frozen = bool(cfg.disable_online_collection)
        if replay_frozen:
            print("[online] disable_online_collection=1 -> fixed replay mode for all rounds")
        for round_idx in range(1, cfg.online_rounds + 1):
            did_collect = False
            if not replay_frozen:
                planner_dataset, planner_stats = collect_planner_dataset(
                    model=model_for_collection(),
                    dataset=current_dataset,
                    env_name=cfg.env,
                    replay_observations=raw_dataset["observations"],
                    replay_timeouts=raw_dataset["timeouts"],
                    horizon=cfg.horizon,
                    device=device,
                    n_episodes=cfg.online_collect_episodes_per_round,
                    episode_len=cfg.online_collect_episode_len,
                    transition_budget=cfg.online_collect_transition_budget_per_round,
                    replan_every_n_steps=cfg.online_replan_every_n_steps,
                    goal_geom_p=cfg.online_goal_geom_p,
                    goal_geom_min_k=cfg.online_goal_geom_min_k,
                    goal_geom_max_k=cfg.online_goal_geom_max_k,
                    goal_min_distance=cfg.online_goal_min_distance,
                    seed=cfg.seed + 10000 + round_idx,
                    maze_arr=maze_arr,
                    wall_aware_planning=cfg.wall_aware_planning,
                    wall_aware_plan_samples=cfg.wall_aware_plan_samples,
                    planning_success_thresholds=online_planning_success_thresholds,
                    planning_success_rel_reduction=cfg.online_planning_success_rel_reduction,
                    early_terminate_on_success=bool(cfg.online_early_terminate_on_success),
                    early_terminate_threshold=float(cfg.online_early_terminate_threshold),
                    min_accepted_episode_len=int(online_min_accepted_episode_len),
                )
                did_collect = True
                raw_dataset = merge_replay_datasets(raw_dataset, planner_dataset)

                if cfg.fixed_replay_snapshot_round > 0 and round_idx == cfg.fixed_replay_snapshot_round:
                    if cfg.fixed_replay_snapshot_npz:
                        snap_path = Path(cfg.fixed_replay_snapshot_npz)
                        if not snap_path.is_absolute():
                            # Interpret relative paths as relative to logdir, unless the user already
                            # included logdir in the provided relative path (common CLI mistake).
                            logdir_parts = tuple(logdir.parts)
                            snap_parts = tuple(snap_path.parts)
                            if logdir_parts and snap_parts[: len(logdir_parts)] == logdir_parts:
                                snap_path = Path(str(snap_path))
                            else:
                                snap_path = logdir / snap_path
                    else:
                        snap_path = logdir / f"replay_snapshot_round{round_idx}.npz"
                    save_replay_npz(
                        path=snap_path,
                        dataset=raw_dataset,
                        action_low=action_low,
                        action_high=action_high,
                        collection_stats=collection_stats,
                        metadata=replay_metadata(stage="snapshot", round_idx=round_idx),
                    )
                    print(f"[replay] snapshot_saved round={round_idx} path={snap_path}")
                    replay_frozen = True
            else:
                # Fixed-replay mode after snapshot: keep training on the same replay without adding new data.
                planner_stats = {
                    "episodes": 0,
                    "episodes_attempted": 0,
                    "episodes_rejected_short": 0,
                    "episode_len_mean": float("nan"),
                    "episode_len_min": float("nan"),
                    "episode_len_max": float("nan"),
                    "transitions": 0,
                    "transitions_attempted": 0,
                    "replans_per_episode_mean": float("nan"),
                    "selected_plan_wall_hits_mean": float("nan"),
                    "rollout_wall_hits_mean": float("nan"),
                    "sampled_goal_distance_mean": float("nan"),
                    "sampled_goal_k_mean": float("nan"),
                }
                print(f"[online-collect] round={round_idx} replay_frozen=1 (skipping collection)")

            replay_transitions = int(len(raw_dataset["observations"]))
            replay_episodes = count_episodes_from_timeouts(raw_dataset["timeouts"])
            round_row = {
                "round": int(round_idx),
                "step_before_retrain": int(global_step),
                "replay_transitions": replay_transitions,
                "replay_episodes": int(replay_episodes),
                "did_collect": int(did_collect),
                "replay_frozen": int(replay_frozen),
                **planner_stats,
            }
            online_collection_rows.append(round_row)
            # Persist online-collection stats incrementally for monitoring/deciders.
            try:
                pd.DataFrame(online_collection_rows).to_csv(logdir / "online_collection.csv", index=False)
            except Exception as e:
                print(f"[warn] failed to flush online_collection.csv: {e}")
            threshold_summaries = " ".join(
                [
                    f"s@{thr:.2f}={planner_stats.get(f'planning_success_rate_final_t{threshold_tag(thr)}', float('nan')):.3f}"
                    for thr in online_planning_success_thresholds
                ]
            )
            rel_tag = threshold_tag(cfg.online_planning_success_rel_reduction)
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
                f"replans_per_ep={planner_stats['replans_per_episode_mean']:.2f} "
                f"plan_wall_hits={planner_stats['selected_plan_wall_hits_mean']:.2f} "
                f"roll_wall_hits={planner_stats['rollout_wall_hits_mean']:.2f} "
                f"rel{cfg.online_planning_success_rel_reduction:.2f}="
                f"{planner_stats.get(f'planning_success_rate_final_rel{rel_tag}', float('nan')):.3f} "
                f"{threshold_summaries}"
            )

            if did_collect:
                current_dataset, train_loader, val_loader, train_idx, val_idx = build_goal_dataset_splits(
                    raw_dataset=raw_dataset,
                    cfg=cfg,
                    split_seed=cfg.seed + 1337 + round_idx,
                    device=device,
                )
                print(
                    "[online-replay] "
                    f"round={round_idx} samples total={len(current_dataset)} "
                    f"train={len(train_idx)} val={len(val_idx)}"
                )
            else:
                print(
                    "[online-replay] "
                    f"round={round_idx} replay_frozen=1 samples total={len(current_dataset)} "
                    f"train={len(train_idx)} val={len(val_idx)}"
                )

            run_training_steps(
                num_steps=cfg.online_train_steps_per_round,
                train_loader_cur=train_loader,
                val_loader_cur=val_loader,
                dataset_for_eval=current_dataset,
                phase=f"online_round_{round_idx}",
            )
    elif cfg.online_self_improve:
        print("[online] enabled but --online_rounds <= 0, skipping replay expansion rounds.")

    replay_export_meta: Dict[str, Any] = {}
    if replay_export_path:
        save_path = Path(replay_export_path)
        if not save_path.is_absolute():
            save_path = logdir / save_path
        replay_export_meta = save_replay_artifact(
            path=save_path,
            dataset=raw_dataset,
            action_low=action_low,
            action_high=action_high,
            collection_stats=collection_stats,
            metadata=replay_metadata(stage="final_export"),
        )
        print(
            "[replay] saved_final "
            f"path={save_path} "
            f"transitions={int(replay_export_meta.get('transitions', len(raw_dataset['observations'])))} "
            f"episodes={int(replay_export_meta.get('episodes', count_episodes_from_timeouts(raw_dataset['timeouts'])))} "
            f"fingerprint={replay_export_meta.get('fingerprint', 'na')}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(logdir / "metrics.csv", index=False)
    plot_losses(metrics_df, out_path=logdir / "train_val_loss.png")

    checkpoint = {
        "step": int(global_step),
        "model": diffusion.state_dict(),
        "ema": ema_model.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, logdir / "checkpoint_last.pt")

    progress_df = pd.DataFrame(progress_rows)
    progress_df.to_csv(logdir / "progress_metrics.csv", index=False)
    if len(online_collection_rows) > 0:
        pd.DataFrame(online_collection_rows).to_csv(logdir / "online_collection.csv", index=False)
    if len(progress_df) > 0:
        plot_progress(progress_df, out_path=logdir / "goal_progress.png")

    query_rows: List[Dict[str, float]] = []
    final_query_pairs = last_eval_query_pairs if len(last_eval_query_pairs) > 0 else query_pairs_for_step(step=global_step)
    query_rollout_env = gym.make(cfg.env)
    query_waypoint_rng = np.random.default_rng(cfg.seed + 424242)
    query_waypoint_t = resolve_waypoint_t(int(cfg.horizon), int(cfg.eval_waypoint_t))
    for qid, (start_xy, goal_xy) in enumerate(final_query_pairs):
        query_waypoint_xy = sample_eval_waypoint(
            mode=cfg.eval_waypoint_mode,
            replay_observations=raw_dataset["observations"],
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoint_eps=float(cfg.eval_waypoint_eps),
            rng=query_waypoint_rng,
        )
        query_waypoint_t_use = query_waypoint_t if query_waypoint_xy is not None else None
        observations, actions = sample_imagined_trajectory(
            model=model_for_eval(),
            dataset=current_dataset,
            start_xy=start_xy,
            goal_xy=goal_xy,
            horizon=cfg.horizon,
            device=device,
            n_samples=cfg.query_batch_size,
            waypoint_xy=query_waypoint_xy,
            waypoint_t=query_waypoint_t_use,
        )

        for sid in range(observations.shape[0]):
            xy = observations[sid, :, :2]
            m = straightness_metrics(xy=xy, start_xy=start_xy, goal_xy=goal_xy)
            b = boundary_jump_ratios(xy)
            imagined_wall_hits = count_wall_hits_qpos_frame(maze_arr, xy)
            rollout_xy, rollout_actions, rollout_min_goal_dist, rollout_final_goal_dist, rollout_wall_hits = rollout_to_goal(
                model=model_for_eval(),
                dataset=current_dataset,
                rollout_env=query_rollout_env,
                start_xy=start_xy,
                goal_xy=goal_xy,
                planning_horizon=cfg.horizon,
                rollout_horizon=cfg.eval_rollout_horizon,
                device=device,
                rollout_mode=cfg.eval_rollout_mode,
                rollout_replan_every_n_steps=cfg.eval_rollout_replan_every_n_steps,
                maze_arr=maze_arr,
                wall_aware_planning=cfg.wall_aware_planning,
                wall_aware_plan_samples=cfg.wall_aware_plan_samples,
                open_loop_actions=actions[sid] if cfg.eval_rollout_mode == "open_loop" else None,
                waypoint_xy=query_waypoint_xy,
                waypoint_t=query_waypoint_t_use,
            )
            waypoint_min_dist = float("nan")
            waypoint_hit = float("nan")
            if query_waypoint_xy is not None:
                wp_d = np.linalg.norm(
                    np.asarray(rollout_xy, dtype=np.float32)[:, :2]
                    - np.asarray(query_waypoint_xy, dtype=np.float32)[None, :],
                    axis=1,
                )
                waypoint_min_dist = float(np.min(wp_d))
                waypoint_hit = float(waypoint_min_dist <= float(cfg.eval_waypoint_eps))
            query_rows.append(
                {
                    "query_id": qid,
                    "sample_id": sid,
                    "start_x": float(start_xy[0]),
                    "start_y": float(start_xy[1]),
                    "goal_x": float(goal_xy[0]),
                    "goal_y": float(goal_xy[1]),
                    "mean_line_deviation": m["mean_line_deviation"],
                    "max_line_deviation": m["max_line_deviation"],
                    "final_goal_error": m["final_goal_error"],
                    "path_length": m["path_length"],
                    "direct_distance": m["direct_distance"],
                    "path_over_direct": m["path_over_direct"],
                    "start_jump_ratio": b["start_jump_ratio"],
                    "end_jump_ratio": b["end_jump_ratio"],
                    "rollout_mode": cfg.eval_rollout_mode,
                    "rollout_min_goal_distance": float(rollout_min_goal_dist),
                    "rollout_final_goal_error": float(rollout_final_goal_dist),
                    "waypoint_x": float(query_waypoint_xy[0]) if query_waypoint_xy is not None else float("nan"),
                    "waypoint_y": float(query_waypoint_xy[1]) if query_waypoint_xy is not None else float("nan"),
                    "waypoint_t": int(query_waypoint_t_use) if query_waypoint_t_use is not None else -1,
                    "waypoint_min_distance": float(waypoint_min_dist),
                    "waypoint_hit": float(waypoint_hit),
                    "imagined_in_wall_points": int(imagined_wall_hits),
                    "rollout_in_wall_points": int(rollout_wall_hits),
                    "xy_json": json.dumps(xy.tolist()),
                    "action_json": json.dumps(actions[sid].tolist()),
                    "rollout_xy_json": json.dumps(rollout_xy.tolist()),
                    "rollout_action_json": json.dumps(rollout_actions.tolist()),
                }
            )
    query_rollout_env.close()

    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(logdir / "query_metrics.csv", index=False)
    plot_query_trajectories(
        query_rows=query_rows[: min(len(query_rows), 8)],
        out_path=logdir / "query_trajectories.png",
        maze_arr=maze_arr,
    )

    summary = {
        "logdir": str(logdir),
        "train_steps_total": int(global_step),
        "online_self_improve": bool(cfg.online_self_improve),
        "online_rounds": int(cfg.online_rounds),
        "online_replan_every_n_steps": int(cfg.online_replan_every_n_steps),
        "collector_weights": str(cfg.collector_weights),
        "eval_weights": str(cfg.eval_weights),
        "collector_ckpt_path": str(cfg.collector_ckpt_path),
        "collector_ckpt_weights": str(cfg.collector_ckpt_weights),
        "replay_import_path": str(replay_import_path),
        "replay_export_path": str(replay_export_path),
        "replay_import_meta": replay_import_meta if replay_import_path else {},
        "replay_export_meta": replay_export_meta if replay_export_path else {},
        "replay_fingerprint": replay_dataset_fingerprint(raw_dataset),
        "fixed_replay_snapshot_round": int(cfg.fixed_replay_snapshot_round),
        "fixed_replay_snapshot_npz": str(cfg.fixed_replay_snapshot_npz),
        "disable_online_collection": bool(cfg.disable_online_collection),
        "dataset_transitions": int(len(raw_dataset["observations"])),
        "dataset_episodes": int(count_episodes_from_timeouts(raw_dataset["timeouts"])),
        "dataset_collection_stats": collection_stats,
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
        "eval_waypoint_mode": str(cfg.eval_waypoint_mode),
        "eval_waypoint_t": int(cfg.eval_waypoint_t),
        "eval_waypoint_eps": float(cfg.eval_waypoint_eps),
        "wall_aware_planning": bool(cfg.wall_aware_planning),
        "wall_aware_plan_samples": int(cfg.wall_aware_plan_samples),
        "online_planning_success_thresholds": [float(thr) for thr in online_planning_success_thresholds],
        "online_planning_success_rel_reduction": float(cfg.online_planning_success_rel_reduction),
        "query_path_over_direct_mean": float(query_df["path_over_direct"].mean()) if len(query_df) else np.nan,
        "query_mean_line_deviation_mean": float(query_df["mean_line_deviation"].mean()) if len(query_df) else np.nan,
        "query_rollout_min_goal_distance_mean": float(query_df["rollout_min_goal_distance"].mean()) if len(query_df) else np.nan,
        "query_rollout_final_goal_error_mean": float(query_df["rollout_final_goal_error"].mean()) if len(query_df) else np.nan,
        "query_waypoint_hit_rate": float(query_df["waypoint_hit"].mean()) if ("waypoint_hit" in query_df.columns and len(query_df)) else np.nan,
        "query_waypoint_min_distance_mean": float(query_df["waypoint_min_distance"].mean()) if ("waypoint_min_distance" in query_df.columns and len(query_df)) else np.nan,
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
