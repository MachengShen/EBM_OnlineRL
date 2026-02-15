#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate a saved checkpoint from scripts/synthetic_maze2d_diffuser_probe.py
and report rollout success at multiple prefix horizons.

Protocol choice:
- Run ONE realized rollout per imagined sample with horizon H_max.
- Compute success@H for multiple prefix horizons from that single rollout trace.
  This guarantees monotonicity (success@64 <= success@128 <= ...), and avoids
  stochastic confounds from re-running separate rollouts per horizon.
"""

import argparse
import copy
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path("/root/ebm-online-rl-prototype")
PROBE = ROOT / "scripts" / "synthetic_maze2d_diffuser_probe.py"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_int_list(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import probe module from {PROBE}")
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    spec.loader.exec_module(probe)
    return probe


def _load_cfg(probe, cfg_path: Path):
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = probe.Config()
    for k, v in raw.items():
        if v is None:
            # Allow older run configs that didn't carry newer fields; keep defaults.
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _build_query_pairs(probe, cfg, raw_dataset: Dict[str, np.ndarray], query_step: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if str(getattr(cfg, "query_mode", "diverse")) == "fixed":
        return list(probe.parse_queries(cfg.query))

    bank_size = int(max(getattr(cfg, "query_bank_size", 256), getattr(cfg, "num_eval_queries", 24)))
    query_bank = probe.build_diverse_query_bank(
        points_xy=raw_dataset["observations"][:, :2],
        bank_size=bank_size,
        n_angle_bins=int(getattr(cfg, "query_angle_bins", 16)),
        min_pair_distance=float(getattr(cfg, "query_min_distance", 1.0)),
        seed=int(cfg.seed) + 991,
    )

    if bool(getattr(cfg, "query_resample_each_eval", True)):
        q_seed = int(cfg.seed) + int(getattr(cfg, "query_resample_seed_stride", 7919)) * max(int(query_step), 1)
    else:
        q_seed = int(cfg.seed) + int(getattr(cfg, "query_resample_seed_stride", 7919))

    return probe.select_query_pairs(
        query_bank=query_bank,
        num_queries=int(getattr(cfg, "num_eval_queries", 24)),
        seed=int(q_seed),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="Run dir containing config.json + checkpoint_last.pt")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path (default: run-dir/checkpoint_last.pt)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--query-step", type=int, default=40000, help="Pseudo-step used only to select the eval query subset.")
    p.add_argument("--num-eval-queries", type=int, default=None)
    p.add_argument("--query-batch-size", type=int, default=None, help="Imagined samples per query.")
    p.add_argument("--goal-success-threshold", type=float, default=None)
    p.add_argument("--eval-rollout-horizon", type=int, default=256)
    p.add_argument("--eval-success-prefix-horizons", type=str, default="64,128,192,256")
    p.add_argument("--eval-rollout-mode", type=str, default="receding_horizon", choices=["open_loop", "receding_horizon"])
    p.add_argument("--eval-rollout-replan-every-n-steps", type=int, default=None)
    p.add_argument("--wall-aware-planning", action="store_true", default=True)
    p.add_argument("--no-wall-aware-planning", dest="wall_aware_planning", action="store_false")
    p.add_argument("--wall-aware-plan-samples", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    probe = _load_probe()

    run_dir: Path = args.run_dir
    ckpt_path = args.checkpoint or (run_dir / "checkpoint_last.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_source = "checkpoint"
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = _load_cfg(probe, cfg_path)
        cfg_source = "config.json"
    else:
        raw = ckpt.get("config", {})
        if not isinstance(raw, dict) or not raw:
            raise SystemExit(f"Missing config.json and checkpoint has no 'config': {ckpt_path}")
        cfg = probe.Config()
        for k, v in raw.items():
            if v is None:
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if args.num_eval_queries is not None:
        cfg.num_eval_queries = int(args.num_eval_queries)
    if args.query_batch_size is not None:
        cfg.query_batch_size = int(args.query_batch_size)
    if args.goal_success_threshold is not None:
        cfg.goal_success_threshold = float(args.goal_success_threshold)
    if args.eval_rollout_replan_every_n_steps is not None:
        cfg.eval_rollout_replan_every_n_steps = int(args.eval_rollout_replan_every_n_steps)

    probe.set_seed(int(cfg.seed))
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] run_dir={run_dir}", flush=True)
    print(f"[setup] checkpoint={ckpt_path}", flush=True)
    print(f"[setup] cfg_source={cfg_source}", flush=True)

    maze_arr = probe.load_maze_arr_from_env(cfg.env)
    raw_dataset, _, _, _ = probe.collect_random_dataset(
        env_name=cfg.env,
        n_episodes=int(cfg.n_episodes),
        episode_len=int(cfg.episode_len),
        action_scale=float(cfg.action_scale),
        seed=int(cfg.seed),
        corridor_aware_data=bool(cfg.corridor_aware_data),
        corridor_max_resamples=int(cfg.corridor_max_resamples),
    )
    dataset, _, _, _, _ = probe.build_goal_dataset_splits(
        raw_dataset=raw_dataset,
        cfg=cfg,
        split_seed=int(cfg.seed) + 1337,
        device=device,
    )

    dim_mults = probe.parse_dim_mults(cfg.model_dim_mults)
    model = probe.TemporalUnet(
        horizon=int(cfg.horizon),
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=dataset.observation_dim,
        dim=int(cfg.model_dim),
        dim_mults=dim_mults,
    ).to(device)
    diffusion = probe.GaussianDiffusion(
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
    ema_model = copy.deepcopy(diffusion).to(device)
    if "ema" not in ckpt:
        raise SystemExit(f"Checkpoint missing 'ema' weights: {ckpt_path}")
    ema_model.load_state_dict(ckpt["ema"])
    ema_model.eval()

    query_pairs = _build_query_pairs(probe, cfg, raw_dataset=raw_dataset, query_step=int(args.query_step))
    prefix_horizons = _parse_int_list(args.eval_success_prefix_horizons)
    if max(prefix_horizons) > int(args.eval_rollout_horizon):
        raise SystemExit("Prefix horizons must be <= eval rollout horizon.")

    print(
        "[eval] "
        f"num_queries={len(query_pairs)} "
        f"samples_per_query={int(cfg.query_batch_size)} "
        f"H_max={int(args.eval_rollout_horizon)} "
        f"prefixes={list(prefix_horizons)} "
        f"threshold={float(cfg.goal_success_threshold):.3f} "
        f"mode={args.eval_rollout_mode} "
        f"replan_every={int(getattr(cfg, 'eval_rollout_replan_every_n_steps', 8))}",
        flush=True,
    )

    metrics = probe.evaluate_goal_progress(
        model=ema_model,
        dataset=dataset,
        env_name=cfg.env,
        query_pairs=query_pairs,
        planning_horizon=int(cfg.horizon),
        rollout_horizon=int(args.eval_rollout_horizon),
        success_prefix_horizons=prefix_horizons,
        device=device,
        n_samples=int(cfg.query_batch_size),
        goal_success_threshold=float(cfg.goal_success_threshold),
        rollout_mode=str(args.eval_rollout_mode),
        rollout_replan_every_n_steps=int(getattr(cfg, "eval_rollout_replan_every_n_steps", 8)),
        maze_arr=maze_arr,
        wall_aware_planning=bool(args.wall_aware_planning),
        wall_aware_plan_samples=int(args.wall_aware_plan_samples),
    )

    # Monotonicity sanity check: since success is defined by prefix-min distance
    # from ONE realized rollout, success@H must be non-decreasing in H.
    succ = []
    for h in prefix_horizons:
        v = float(metrics.get(f"rollout_goal_success_rate_h{int(h)}", float("nan")))
        succ.append((int(h), v))
    mono_ok = True
    for i in range(len(succ) - 1):
        if not (succ[i][1] <= succ[i + 1][1] + 1e-12):
            mono_ok = False
            break

    out_dir = args.out_dir or (run_dir / f"eval_prefix_{_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "query_step": int(args.query_step),
        "cfg_eval_overrides": {
            "num_eval_queries": int(getattr(cfg, "num_eval_queries", 24)),
            "query_batch_size": int(getattr(cfg, "query_batch_size", 6)),
            "goal_success_threshold": float(getattr(cfg, "goal_success_threshold", 0.5)),
            "eval_rollout_horizon": int(args.eval_rollout_horizon),
            "eval_success_prefix_horizons": [int(h) for h in prefix_horizons],
            "eval_rollout_mode": str(args.eval_rollout_mode),
            "eval_rollout_replan_every_n_steps": int(getattr(cfg, "eval_rollout_replan_every_n_steps", 8)),
            "wall_aware_planning": bool(args.wall_aware_planning),
            "wall_aware_plan_samples": int(args.wall_aware_plan_samples),
        },
        "monotonic_success_ok": bool(mono_ok),
        "success_by_prefix": [{"h": int(h), "success": float(v)} for h, v in succ],
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in metrics.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'metrics.json'}", flush=True)
    print("[done] success_by_prefix:", succ, "mono_ok=", mono_ok, flush=True)


if __name__ == "__main__":
    main()
