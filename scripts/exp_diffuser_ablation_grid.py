#!/usr/bin/env python3
"""
Diffuser execution-time ablation grid.

Sweeps (alpha, beta, adaptive_replan, plan_samples) using
analyze_collector_stochasticity.py as the evaluation harness.
Saves per-condition JSON summaries and a merged CSV.

Usage:
  python3 scripts/exp_diffuser_ablation_grid.py \
    --diffuser_run_dir <path/to/diffuser/seed_N> \
    --sac_run_dir     <path/to/sac/seed_N> \
    --base_dir runs/analysis/ablation_grid/grid_TIMESTAMP
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

PYTHON = "third_party/diffuser/.venv38/bin/python3.8"
ENV_PREFIX = (
    "D4RL_SUPPRESS_IMPORT_ERROR=1 "
    "MUJOCO_GL=egl "
    "LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH "
    "PYTHONPATH=third_party/diffuser-maze2d"
)
ANALYZER = "scripts/analyze_collector_stochasticity.py"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diffuser execution-time ablation grid.")
    p.add_argument("--diffuser_run_dir", required=True,
                   help="Path to a Diffuser phase1 run dir (contains config.json + checkpoint).")
    p.add_argument("--sac_run_dir", default="",
                   help="Optional SAC run dir for baseline comparison.")
    p.add_argument("--base_dir", default=f"runs/analysis/ablation_grid/grid_{_stamp()}",
                   help="Output root directory.")
    p.add_argument("--device", default="cuda:0")
    # Eval budget (same as visual_check defaults for fast feedback)
    p.add_argument("--num_queries", type=int, default=6)
    p.add_argument("--samples_per_query", type=int, default=20)
    p.add_argument("--rollouts_per_query", type=int, default=6)
    p.add_argument("--rollout_horizon", type=int, default=192)
    p.add_argument("--diffuser_replan_every", type=int, default=8)
    p.add_argument("--goal_success_threshold", type=float, default=0.5)
    # Grid axes
    p.add_argument("--alpha_grid", default="1.0,1.2,1.4",
                   help="Comma-separated action_scale_mult values.")
    p.add_argument("--beta_grid", default="0.0,0.5",
                   help="Comma-separated action_ema_beta values.")
    p.add_argument("--adaptive_grid", default="0,1",
                   help="0=no adaptive, 1=adaptive. Comma-separated.")
    p.add_argument("--plan_samples_grid", default="1",
                   help="Comma-separated plan_samples values (1=no scoring, 2+=best-of-K).")
    p.add_argument("--plan_score_mode", default="min_dist_prefix",
                   help="Scoring mode when plan_samples>1.")
    return p.parse_args()


def run_condition(
    args: argparse.Namespace,
    *,
    alpha: float,
    beta: float,
    adaptive: bool,
    plan_samples: int,
    out_dir: Path,
) -> dict:
    """Run one ablation condition via subprocess and return its summary dict."""
    cond_name = (
        f"alpha{alpha:.2f}_beta{beta:.2f}_adapt{int(adaptive)}_K{plan_samples}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cmd_parts = [
        ENV_PREFIX,
        PYTHON,
        ANALYZER,
        f"--diffuser_run_dir {args.diffuser_run_dir}",
        f"--num_queries {args.num_queries}",
        f"--samples_per_query {args.samples_per_query}",
        f"--rollouts_per_query {args.rollouts_per_query}",
        f"--rollout_horizon {args.rollout_horizon}",
        f"--diffuser_replan_every {args.diffuser_replan_every}",
        f"--goal_success_threshold {args.goal_success_threshold}",
        f"--device {args.device}",
        f"--diffuser-action-scale-mult {alpha}",
        f"--diffuser-action-ema-beta {beta}",
        "--adaptive-replan" if adaptive else "--no-adaptive-replan",
        f"--plan-samples {plan_samples}",
        f"--plan-score-mode {args.plan_score_mode if plan_samples > 1 else 'none'}",
        f"--outdir {out_dir}",
    ]
    if args.sac_run_dir:
        cmd_parts.append(f"--sac_run_dir {args.sac_run_dir}")

    cmd = " ".join(cmd_parts)
    print(f"\n[GRID] {cond_name}")
    print(f"  cmd: {cmd[:120]}...")

    with open(log_path, "w") as log_f:
        ret = subprocess.run(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT, check=False)

    summary_path = out_dir / "collector_stochasticity_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
    else:
        data = {}
        print(f"  [WARN] no summary (exit={ret.returncode}); see {log_path}")

    data["_condition"] = cond_name
    data["_alpha"] = alpha
    data["_beta"] = beta
    data["_adaptive_replan"] = int(adaptive)
    data["_plan_samples"] = plan_samples
    data["_exit_code"] = ret.returncode

    # Print quick readout
    succ = data.get("diffuser_rollout_success_rate_mean", float("nan"))
    mgd = data.get("diffuser_rollout_min_goal_dist_mean", float("nan"))
    print(f"  => success={succ:.4f}  min_goal_dist={mgd:.4f}")

    return data


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    alphas = [float(x) for x in args.alpha_grid.split(",")]
    betas = [float(x) for x in args.beta_grid.split(",")]
    adapt_vals = [bool(int(x)) for x in args.adaptive_grid.split(",")]
    plan_vals = [int(x) for x in args.plan_samples_grid.split(",")]

    total = len(alphas) * len(betas) * len(adapt_vals) * len(plan_vals)
    print(f"[GRID] Starting ablation grid: {total} conditions")
    print(f"[GRID] alpha={alphas}  beta={betas}  adaptive={adapt_vals}  plan_samples={plan_vals}")
    print(f"[GRID] Output: {base}")

    results = []
    for alpha, beta, adaptive, plan_samples in product(alphas, betas, adapt_vals, plan_vals):
        cond_name = f"alpha{alpha:.2f}_beta{beta:.2f}_adapt{int(adaptive)}_K{plan_samples}"
        out_dir = base / cond_name
        data = run_condition(
            args,
            alpha=alpha,
            beta=beta,
            adaptive=adaptive,
            plan_samples=plan_samples,
            out_dir=out_dir,
        )
        results.append(data)

    # Save merged CSV
    merged_path = base / "ablation_grid_results.csv"
    if results:
        all_keys = []
        for r in results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(merged_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in all_keys})
    print(f"\n[GRID] Done. Merged results: {merged_path}")

    # Print top-3 by success rate
    ranked = sorted(
        [r for r in results if isinstance(r.get("diffuser_rollout_success_rate_mean"), float)],
        key=lambda r: r.get("diffuser_rollout_success_rate_mean", 0.0),
        reverse=True,
    )
    print("\n[GRID] Top-3 conditions by diffuser success rate:")
    for r in ranked[:3]:
        print(
            f"  {r['_condition']}: "
            f"success={r.get('diffuser_rollout_success_rate_mean', float('nan')):.4f}  "
            f"min_dist={r.get('diffuser_rollout_min_goal_dist_mean', float('nan')):.4f}  "
            f"hit_0p1={r.get('diffuser_hit_rate_0p1', float('nan')):.4f}"
        )

    # Print top-3 by min_goal_dist (lower is better)
    ranked_dist = sorted(
        [r for r in results if isinstance(r.get("diffuser_rollout_min_goal_dist_mean"), float)],
        key=lambda r: r.get("diffuser_rollout_min_goal_dist_mean", float("inf")),
    )
    print("\n[GRID] Top-3 by min_goal_dist (lower = better):")
    for r in ranked_dist[:3]:
        print(
            f"  {r['_condition']}: "
            f"min_dist={r.get('diffuser_rollout_min_goal_dist_mean', float('nan')):.4f}  "
            f"success={r.get('diffuser_rollout_success_rate_mean', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
