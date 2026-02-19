#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/ebm-online-rl-prototype")
PROBE = ROOT / "scripts" / "synthetic_maze2d_diffuser_probe.py"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _build_query_pairs(probe, cfg, raw_dataset: Dict[str, np.ndarray], query_step: int, num_queries: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if str(getattr(cfg, "query_mode", "diverse")) == "fixed":
        pairs = list(probe.parse_queries(cfg.query))
        return pairs[:num_queries]

    bank_size = int(max(getattr(cfg, "query_bank_size", 256), num_queries))
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
        num_queries=num_queries,
        seed=int(q_seed),
    )


def _mode_signature(xy: np.ndarray, cell_size: float) -> str:
    cells = np.floor(np.asarray(xy, dtype=np.float32) / float(cell_size)).astype(np.int32)
    uniq = np.unique(cells, axis=0)
    # Cap to keep signatures bounded in report size.
    if len(uniq) > 128:
        uniq = uniq[:128]
    return "|".join(f"{int(a)}:{int(b)}" for a, b in uniq)


def _pairwise_distances(xy: np.ndarray) -> np.ndarray:
    # xy: [K, T, 2]
    k = int(xy.shape[0])
    if k < 2:
        return np.asarray([], dtype=np.float32)
    diff = xy[:, None, :, :] - xy[None, :, :, :]
    d = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    tri = np.triu_indices(k, 1)
    return np.asarray(d[tri], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline posterior-diversity diagnostics for a trained diffuser checkpoint.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--query-step", type=int, default=40000)
    ap.add_argument("--num-queries", type=int, default=12)
    ap.add_argument("--samples-per-query", type=int, default=50)
    ap.add_argument("--planning-horizon", type=int, default=None)
    ap.add_argument("--goal-success-threshold", type=float, default=None)
    ap.add_argument("--cell-size", type=float, default=0.25)
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    probe = _load_probe()

    # Ensure diffuser/MuJoCo env vars are present.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

    run_dir = args.run_dir.resolve()
    ckpt_path = args.checkpoint.resolve() if args.checkpoint else (run_dir / "checkpoint_last.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config.json: {cfg_path}")
    cfg = _load_cfg(probe, cfg_path)
    if args.goal_success_threshold is not None:
        cfg.goal_success_threshold = float(args.goal_success_threshold)

    planning_horizon = int(args.planning_horizon) if args.planning_horizon is not None else int(cfg.horizon)
    if planning_horizon <= 0:
        raise SystemExit("planning horizon must be > 0")
    if args.samples_per_query <= 0:
        raise SystemExit("samples-per-query must be > 0")

    probe.set_seed(int(cfg.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

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

    query_pairs = _build_query_pairs(
        probe,
        cfg,
        raw_dataset=raw_dataset,
        query_step=int(args.query_step),
        num_queries=int(args.num_queries),
    )

    rows: List[Dict[str, float | int | str]] = []
    for qid, (start_xy, goal_xy) in enumerate(query_pairs):
        observations, _actions = probe.sample_imagined_trajectory(
            model=ema_model,
            dataset=dataset,
            start_xy=start_xy,
            goal_xy=goal_xy,
            horizon=int(planning_horizon),
            device=device,
            n_samples=int(args.samples_per_query),
        )
        xy = np.asarray(observations[:, :, :2], dtype=np.float32)
        final_err = np.linalg.norm(xy[:, -1, :] - np.asarray(goal_xy, dtype=np.float32)[None, :], axis=1)
        min_goal = np.linalg.norm(
            xy - np.asarray(goal_xy, dtype=np.float32)[None, None, :],
            axis=2,
        ).min(axis=1)
        success_final = float(np.mean(final_err <= float(cfg.goal_success_threshold)))
        success_prefix = float(np.mean(min_goal <= float(cfg.goal_success_threshold)))

        pair = _pairwise_distances(xy)
        pair_mean = float(np.mean(pair)) if pair.size else float("nan")
        pair_p90 = float(np.percentile(pair, 90)) if pair.size else float("nan")

        mode_counts: Dict[str, int] = {}
        for i in range(xy.shape[0]):
            key = _mode_signature(xy[i], cell_size=float(args.cell_size))
            mode_counts[key] = mode_counts.get(key, 0) + 1
        mode_count = int(len(mode_counts))
        top_mode_frac = float(max(mode_counts.values()) / xy.shape[0]) if mode_counts else float("nan")

        rows.append(
            {
                "query_id": int(qid),
                "start_x": float(start_xy[0]),
                "start_y": float(start_xy[1]),
                "goal_x": float(goal_xy[0]),
                "goal_y": float(goal_xy[1]),
                "samples": int(args.samples_per_query),
                "pairwise_mean_distance": float(pair_mean),
                "pairwise_p90_distance": float(pair_p90),
                "mode_count": int(mode_count),
                "top_mode_fraction": float(top_mode_frac),
                "imagined_success_final": float(success_final),
                "imagined_success_prefix": float(success_prefix),
            }
        )

    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / f"posterior_diversity_{_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "posterior_diversity.csv"
    df.to_csv(csv_path, index=False)

    report_lines: List[str] = []
    report_lines.append("Posterior Diversity Report")
    report_lines.append(f"Created: {datetime.now().isoformat()}")
    report_lines.append(f"Run dir: {run_dir}")
    report_lines.append(f"Checkpoint: {ckpt_path}")
    report_lines.append(f"Queries: {len(df)}")
    report_lines.append(f"Samples/query: {int(args.samples_per_query)}")
    report_lines.append("")
    if len(df) > 0:
        report_lines.append(f"Mean pairwise distance: {float(df['pairwise_mean_distance'].mean()):.4f}")
        report_lines.append(f"Mean mode count: {float(df['mode_count'].mean()):.2f}")
        report_lines.append(f"Mean top-mode fraction: {float(df['top_mode_fraction'].mean()):.4f}")
        report_lines.append(
            f"Mean imagined success (final): {float(df['imagined_success_final'].mean()):.4f}"
        )
        report_lines.append(
            f"Mean imagined success (prefix): {float(df['imagined_success_prefix'].mean()):.4f}"
        )
        report_lines.append("")
        ranked = df.sort_values("mode_count", ascending=False).head(3)
        report_lines.append("Top-3 queries by mode_count:")
        for _, r in ranked.iterrows():
            report_lines.append(
                "  - query_id={qid} mode_count={m} pairwise_mean={p:.4f} success_final={s:.4f}".format(
                    qid=int(r["query_id"]),
                    m=int(r["mode_count"]),
                    p=float(r["pairwise_mean_distance"]),
                    s=float(r["imagined_success_final"]),
                )
            )

    report_path = out_dir / "posterior_diversity_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[done] csv={csv_path}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
