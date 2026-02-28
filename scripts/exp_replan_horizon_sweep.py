#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_csv_int(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def _base_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    mj = "/root/.mujoco/mujoco210/bin"
    ld = env.get("LD_LIBRARY_PATH", "")
    if mj not in ld.split(":"):
        env["LD_LIBRARY_PATH"] = f"{ld}:{mj}" if ld else mj
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    pp = str(root / "third_party" / "diffuser-maze2d")
    cur_pp = env.get("PYTHONPATH", "")
    if pp not in cur_pp.split(":"):
        env["PYTHONPATH"] = f"{cur_pp}:{pp}" if cur_pp else pp
    return env


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path, timeout_sec: int) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        f.flush()
        try:
            cp = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_sec, check=False)
            rc = int(cp.returncode)
            status = "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            rc = 124
            status = "timeout"
            f.write(f"\n[timeout] exceeded {timeout_sec}s\n")
    return {"rc": rc, "status": status, "wallclock_sec": float(time.time() - t0), "log_path": str(log_path)}


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Evaluate checkpoint over replan-frequency x planning-horizon sweep.")
    ap.add_argument("--root", type=Path, default=root_default)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--replan-values", type=str, default="1,4,8,16,32,64,256")
    ap.add_argument("--horizon-values", type=str, default="16,32,64,128")
    ap.add_argument("--eval-rollout-horizon", type=int, default=256)
    ap.add_argument("--prefix-horizons", type=str, default="64,128,192,256")
    ap.add_argument("--num-eval-queries", type=int, default=8)
    ap.add_argument("--query-batch-size", type=int, default=2)
    ap.add_argument("--goal-success-threshold", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--timeout-min", type=float, default=90.0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    py = args.python if args.python.is_absolute() else (Path.cwd() / args.python)
    run_dir = args.run_dir.resolve()
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")
    if not py.exists():
        raise SystemExit(f"Missing python: {py}")
    if not run_dir.exists():
        raise SystemExit(f"Missing run dir: {run_dir}")

    replan_values = _parse_csv_int(args.replan_values)
    horizon_values = _parse_csv_int(args.horizon_values)
    prefix_horizons = _parse_csv_int(args.prefix_horizons)
    if args.smoke:
        replan_values = replan_values[:2]
        horizon_values = horizon_values[:2]
        prefix_horizons = prefix_horizons[:2]

    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / f"replan_horizon_sweep_{_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_script = root / "scripts" / "eval_synth_maze2d_checkpoint_prefix.py"
    if not eval_script.exists():
        raise SystemExit(f"Missing eval script: {eval_script}")

    env = _base_env(root)
    timeout_sec = int(float(args.timeout_min) * 60.0)
    rows: List[Dict[str, Any]] = []

    for horizon in horizon_values:
        for replan_every in replan_values:
            cell = out_dir / f"h{int(horizon)}_r{int(replan_every)}"
            cell.mkdir(parents=True, exist_ok=True)
            metrics_path = cell / "metrics.json"
            cmd = [
                str(py),
                str(eval_script),
                "--run-dir",
                str(run_dir),
                "--device",
                str(args.device),
                "--planning-horizon",
                str(horizon),
                "--eval-rollout-replan-every-n-steps",
                str(replan_every),
                "--eval-rollout-horizon",
                str(args.eval_rollout_horizon),
                "--eval-success-prefix-horizons",
                ",".join(str(x) for x in prefix_horizons),
                "--num-eval-queries",
                str(args.num_eval_queries),
                "--query-batch-size",
                str(args.query_batch_size),
                "--goal-success-threshold",
                str(args.goal_success_threshold),
                "--out-dir",
                str(cell),
            ]
            if args.checkpoint is not None:
                cmd.extend(["--checkpoint", str(args.checkpoint.resolve())])

            run_result = _run(cmd, cwd=root, env=env, log_path=cell / "stdout_stderr.log", timeout_sec=timeout_sec)
            row: Dict[str, Any] = {
                "planning_horizon": int(horizon),
                "eval_replan_every_n_steps": int(replan_every),
                **run_result,
                "metrics_path": str(metrics_path),
            }
            if metrics_path.exists():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
                except Exception:
                    metrics = {}
                for h in prefix_horizons:
                    h_int = int(h)
                    row[f"success_at_{h_int}"] = float(metrics.get(f"rollout_goal_success_rate_h{h_int}", np.nan))
                    row[f"goal_query_coverage_at_{h_int}"] = float(
                        metrics.get(f"rollout_goal_query_coverage_rate_h{h_int}", np.nan)
                    )
                    row[f"goal_cell_coverage_at_{h_int}"] = float(
                        metrics.get(f"rollout_goal_cell_coverage_rate_h{h_int}", np.nan)
                    )
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "replan_horizon_sweep.csv"
    df.to_csv(csv_path, index=False)

    md_lines: List[str] = ["# Replan x Horizon Sweep", "", f"Rows: {len(df)}", ""]
    if not df.empty:
        target_h = int(prefix_horizons[-1])
        succ_col = f"success_at_{target_h}"
        ok = df[df["rc"] == 0]
        if not ok.empty and succ_col in ok.columns:
            best_idx = ok[succ_col].idxmax()
            worst_idx = ok[succ_col].idxmin()
            best = ok.loc[best_idx]
            worst = ok.loc[worst_idx]
            md_lines.append(
                f"Best (success@{target_h}): horizon={int(best['planning_horizon'])}, replan={int(best['eval_replan_every_n_steps'])}, value={float(best[succ_col]):.4f}"
            )
            md_lines.append(
                f"Worst (success@{target_h}): horizon={int(worst['planning_horizon'])}, replan={int(worst['eval_replan_every_n_steps'])}, value={float(worst[succ_col]):.4f}"
            )
            md_lines.append("")

    md_path = out_dir / "replan_horizon_sweep.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
