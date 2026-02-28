#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import gym
import numpy as np
import torch

import d4rl  # noqa: F401


ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = ROOT / "scripts" / "synthetic_maze2d_diffuser_probe.py"
UTILS_PATH = ROOT / "scripts" / "maze2d_eqm_utils.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_int_list(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _build_query_pairs(
    probe,
    raw_dataset: Mapping[str, np.ndarray],
    *,
    num_eval_queries: int,
    query_bank_size: int,
    query_angle_bins: int,
    query_min_distance: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    bank_size = int(max(query_bank_size, num_eval_queries))
    query_bank = probe.build_diverse_query_bank(
        points_xy=np.asarray(raw_dataset["observations"], dtype=np.float32)[:, :2],
        bank_size=bank_size,
        n_angle_bins=int(query_angle_bins),
        min_pair_distance=float(query_min_distance),
        seed=int(seed) + 991,
    )
    return probe.select_query_pairs(
        query_bank=query_bank,
        num_queries=int(num_eval_queries),
        seed=int(seed) + 7919,
    )


def _resolve_scaffold_cfg(args: argparse.Namespace) -> Dict[str, Any] | None:
    if str(args.scaffold) == "none":
        return None
    return {
        "enabled": True,
        "stride": int(args.scaffold_stride),
        "insert_step": int(args.scaffold_insert_step) if args.scaffold_insert_step is not None else None,
        "insert_frac": float(args.scaffold_insert_frac),
        "anchor_mode": "pos_only_xy",
        "exclude_endpoints": True,
        "log_insert": bool(args.scaffold_log_insert),
    }


def _safe_mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


@torch.no_grad()
def _compute_smoothness_metrics(
    probe,
    model,
    dataset,
    query_pairs: List[Tuple[np.ndarray, np.ndarray]],
    *,
    planning_horizon: int,
    device: torch.device,
    n_samples: int,
    sampling_cfg: Mapping[str, Any] | None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    scaffold_cfg = sampling_cfg.get("scaffold_cfg") if sampling_cfg else None
    diff_steps_override = sampling_cfg.get("diffusion_steps_override") if sampling_cfg else None
    eqm_eta_start = sampling_cfg.get("eqm_eta_start") if sampling_cfg else None
    eqm_eta_end = sampling_cfg.get("eqm_eta_end") if sampling_cfg else None

    pos_smooth_vals: List[float] = []
    act_smooth_vals: List[float] = []
    pos_rough_vals: List[float] = []
    rows: List[Dict[str, float]] = []

    for qid, (start_xy, goal_xy) in enumerate(query_pairs):
        observations, actions = probe.sample_imagined_trajectory(
            model=model,
            dataset=dataset,
            start_xy=start_xy,
            goal_xy=goal_xy,
            horizon=int(planning_horizon),
            device=device,
            n_samples=int(n_samples),
            waypoint_xy=None,
            waypoint_t=None,
            scaffold_cfg=scaffold_cfg,
            diffusion_steps_override=diff_steps_override,
            eqm_eta_start=eqm_eta_start,
            eqm_eta_end=eqm_eta_end,
        )
        for sid in range(int(observations.shape[0])):
            pos = np.asarray(observations[sid, :, :2], dtype=np.float32)
            act = np.asarray(actions[sid, :, :], dtype=np.float32)

            if pos.shape[0] > 1:
                smooth_pos = float(np.mean(np.linalg.norm(pos[1:] - pos[:-1], axis=-1)))
                smooth_act = float(np.mean(np.linalg.norm(act[1:] - act[:-1], axis=-1)))
            else:
                smooth_pos = float("nan")
                smooth_act = float("nan")
            if pos.shape[0] > 2:
                rough_pos = float(
                    np.mean(
                        np.linalg.norm(
                            pos[2:] - 2.0 * pos[1:-1] + pos[:-2],
                            axis=-1,
                        )
                    )
                )
            else:
                rough_pos = float("nan")

            pos_smooth_vals.append(smooth_pos)
            act_smooth_vals.append(smooth_act)
            pos_rough_vals.append(rough_pos)
            rows.append(
                {
                    "query_id": float(qid),
                    "sample_id": float(sid),
                    "smooth_pos": smooth_pos,
                    "smooth_act": smooth_act,
                    "rough_pos": rough_pos,
                }
            )

    smooth_pos_mean, smooth_pos_std = _safe_mean_std(pos_smooth_vals)
    smooth_act_mean, smooth_act_std = _safe_mean_std(act_smooth_vals)
    rough_pos_mean, rough_pos_std = _safe_mean_std(pos_rough_vals)
    metrics = {
        "smooth_pos_mean": smooth_pos_mean,
        "smooth_pos_std": smooth_pos_std,
        "smooth_act_mean": smooth_act_mean,
        "smooth_act_std": smooth_act_std,
        "rough_pos_mean": rough_pos_mean,
        "rough_pos_std": rough_pos_std,
        "smoothness_num_trajectories": float(len(rows)),
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Maze2D scaffold probe for EqM vs Diffuser checkpoints.")
    p.add_argument("--algo", type=str, required=True, choices=["eqm", "diffuser"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--env", type=str, default="", help="Override env name. Default: checkpoint config env.")

    p.add_argument("--planning_horizon", type=int, default=64)
    p.add_argument("--rollout_horizon", type=int, default=256)
    p.add_argument("--rollout_mode", type=str, default="receding_horizon", choices=["open_loop", "receding_horizon"])
    p.add_argument("--rollout_replan_every", type=int, default=8)
    p.add_argument("--goal_success_threshold", type=float, default=0.5)

    p.add_argument("--num_eval_queries", type=int, default=8)
    p.add_argument("--query_batch_size", type=int, default=2)
    p.add_argument("--query_bank_size", type=int, default=256)
    p.add_argument("--query_angle_bins", type=int, default=16)
    p.add_argument("--query_min_distance", type=float, default=1.0)

    p.add_argument("--scaffold", type=str, default="none", choices=["none", "insert_mid"])
    p.add_argument("--scaffold_stride", type=int, default=8)
    p.add_argument("--scaffold_insert_frac", type=float, default=0.3)
    p.add_argument("--scaffold_insert_step", type=int, default=None)
    p.add_argument("--scaffold_log_insert", action="store_true", default=False)

    p.add_argument("--eqm_steps", type=int, default=None)
    p.add_argument("--eqm_eta", type=float, default=None)
    p.add_argument("--eqm_eta_start", type=float, default=None)
    p.add_argument("--eqm_eta_end", type=float, default=None)

    p.add_argument("--diff_steps", type=int, default=None, help="Override diffusion denoising steps (<= checkpoint n_timesteps).")
    p.add_argument("--eval_prefixes", type=str, default="64,128,192,256")

    p.add_argument("--wall_aware_planning", action="store_true", default=False)
    p.add_argument("--wall_aware_plan_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    probe = _load_module(PROBE_PATH, "maze2d_probe_mod")
    mutils = _load_module(UTILS_PATH, "maze2d_utils_mod")

    if args.wall_aware_plan_samples <= 0:
        raise SystemExit("--wall_aware_plan_samples must be > 0")
    if args.query_batch_size <= 0:
        raise SystemExit("--query_batch_size must be > 0")
    if args.planning_horizon <= 0:
        raise SystemExit("--planning_horizon must be > 0")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cpu":
        print("[warn] CUDA unavailable; running probe on CPU.", flush=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}))
    env_name = str(args.env) if str(args.env) else str(cfg.get("env", "maze2d-umaze-v1"))

    if args.algo == "eqm":
        model, dataset, loaded_cfg = mutils.load_eqm_model_and_dataset(str(args.checkpoint), device)
        if args.eqm_steps is not None:
            model.n_eqm_steps = int(args.eqm_steps)
        if args.eqm_eta is not None:
            model.step_size = float(args.eqm_eta)
        base_eta = float(model.step_size)
        eqm_eta_start = float(args.eqm_eta_start) if args.eqm_eta_start is not None else None
        eqm_eta_end = float(args.eqm_eta_end) if args.eqm_eta_end is not None else None
        if eqm_eta_start is not None and eqm_eta_end is None:
            eqm_eta_end = base_eta
        if eqm_eta_end is not None and eqm_eta_start is None:
            eqm_eta_start = base_eta
        diffusion_steps_override = None
    else:
        model, dataset, loaded_cfg = mutils.load_diffuser_model_and_dataset(str(args.checkpoint), device)
        eqm_eta_start = None
        eqm_eta_end = None
        diffusion_steps_override = int(args.diff_steps) if args.diff_steps is not None else None
        if diffusion_steps_override is not None and diffusion_steps_override > int(model.n_timesteps):
            raise SystemExit(
                f"--diff_steps={diffusion_steps_override} exceeds checkpoint n_timesteps={int(model.n_timesteps)}"
            )

    planning_horizon = int(args.planning_horizon)
    eval_prefixes = _parse_int_list(args.eval_prefixes)
    if max(eval_prefixes) > int(args.rollout_horizon):
        raise SystemExit("--eval_prefixes values must be <= --rollout_horizon")

    env = gym.make(env_name)
    raw_dataset = env.get_dataset()
    env.close()

    query_pairs = _build_query_pairs(
        probe,
        raw_dataset=raw_dataset,
        num_eval_queries=int(args.num_eval_queries),
        query_bank_size=int(args.query_bank_size),
        query_angle_bins=int(args.query_angle_bins),
        query_min_distance=float(args.query_min_distance),
        seed=int(args.seed),
    )
    maze_arr = probe.load_maze_arr_from_env(env_name)

    scaffold_cfg = _resolve_scaffold_cfg(args)
    sampling_cfg = {
        "scaffold_cfg": scaffold_cfg,
        "diffusion_steps_override": diffusion_steps_override,
        "eqm_eta_start": eqm_eta_start,
        "eqm_eta_end": eqm_eta_end,
    }

    metrics = probe.evaluate_goal_progress(
        model=model,
        dataset=dataset,
        env_name=env_name,
        replay_observations=np.asarray(raw_dataset["observations"], dtype=np.float32),
        query_pairs=query_pairs,
        planning_horizon=planning_horizon,
        rollout_horizon=int(args.rollout_horizon),
        success_prefix_horizons=eval_prefixes,
        device=device,
        n_samples=int(args.query_batch_size),
        goal_success_threshold=float(args.goal_success_threshold),
        rollout_mode=str(args.rollout_mode),
        rollout_replan_every_n_steps=int(args.rollout_replan_every),
        maze_arr=maze_arr,
        wall_aware_planning=bool(args.wall_aware_planning),
        wall_aware_plan_samples=int(args.wall_aware_plan_samples),
        eval_waypoint_mode="none",
        eval_waypoint_t=0,
        eval_waypoint_eps=float(args.goal_success_threshold),
        sampling_cfg=sampling_cfg,
    )

    smoothness_metrics, smoothness_rows = _compute_smoothness_metrics(
        probe=probe,
        model=model,
        dataset=dataset,
        query_pairs=query_pairs,
        planning_horizon=planning_horizon,
        device=device,
        n_samples=int(args.query_batch_size),
        sampling_cfg=sampling_cfg,
    )

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            ROOT
            / "runs"
            / "analysis"
            / f"maze2d_scaffold_probe_{_stamp()}"
            / f"{args.algo}_{args.scaffold}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    smoothness_csv = out_dir / "smoothness_samples.csv"
    with smoothness_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "sample_id", "smooth_pos", "smooth_act", "rough_pos"],
        )
        writer.writeheader()
        for row in smoothness_rows:
            writer.writerow(row)

    summary = {
        "created_at": datetime.now().isoformat(),
        "algo": str(args.algo),
        "checkpoint": str(args.checkpoint),
        "env": env_name,
        "device": str(device),
        "loaded_cfg_horizon": int(loaded_cfg.get("horizon", planning_horizon)),
        "planning_horizon": planning_horizon,
        "rollout_horizon": int(args.rollout_horizon),
        "rollout_mode": str(args.rollout_mode),
        "rollout_replan_every": int(args.rollout_replan_every),
        "goal_success_threshold": float(args.goal_success_threshold),
        "num_eval_queries": int(args.num_eval_queries),
        "query_batch_size": int(args.query_batch_size),
        "eval_prefixes": [int(x) for x in eval_prefixes],
        "sampling_cfg": sampling_cfg,
        "eval_metrics": metrics,
        "smoothness_metrics": smoothness_metrics,
        "artifacts": {
            "smoothness_samples_csv": str(smoothness_csv),
        },
    }
    summary_path = out_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[done] wrote {summary_path}", flush=True)
    print(
        "[summary] "
        f"rollout_success={float(metrics.get('rollout_goal_success_rate', float('nan'))):.4f} "
        f"smooth_pos={float(smoothness_metrics.get('smooth_pos_mean', float('nan'))):.4f} "
        f"smooth_act={float(smoothness_metrics.get('smooth_act_mean', float('nan'))):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
