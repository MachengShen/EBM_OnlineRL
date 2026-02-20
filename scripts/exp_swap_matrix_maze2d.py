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
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_COLLECTORS = ("diffuser", "sac_her_sparse")
DEFAULT_LEARNERS = ("diffuser", "sac_her_sparse")
DEFAULT_MODES = ("warmstart", "frozen")
DEFAULT_SEEDS = (0, 1, 2)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_csv(raw: str) -> List[str]:
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one comma-separated token")
    return out


def _parse_csv_int(raw: str) -> List[int]:
    return [int(x) for x in _parse_csv(raw)]


def _default_python(root: Path) -> Path:
    candidates = [
        root / "third_party" / "diffuser" / ".venv38" / "bin" / "python3",
        root / ".venv" / "bin" / "python3",
        Path(sys.executable),
    ]
    for p in candidates:
        if p.exists() and os.access(str(p), os.X_OK):
            return p
    return Path(sys.executable)


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


def _method_script(root: Path, name: str) -> Path:
    if name == "diffuser":
        return root / "scripts" / "synthetic_maze2d_diffuser_probe.py"
    if name == "sac_her_sparse":
        return root / "scripts" / "synthetic_maze2d_sac_her_probe.py"
    raise ValueError(f"Unsupported method: {name}")


def _method_extra_args(name: str) -> List[str]:
    if name == "diffuser":
        return []
    if name == "sac_her_sparse":
        return ["--reward_mode", "sparse"]
    raise ValueError(f"Unsupported method: {name}")


def _dict_to_cli(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in d.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
            else:
                args.append(f"--no_{k}")
        else:
            args.extend([flag, str(v)])
    return args


def _run_cmd(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str], log_path: Path, timeout_sec: int) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        f.flush()
        try:
            cp = subprocess.run(
                list(cmd),
                cwd=str(cwd),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            rc = int(cp.returncode)
            status = "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            rc = 124
            status = "timeout"
            f.write(f"\n[timeout] exceeded {timeout_sec}s\n")
    ended = time.time()
    return {
        "rc": int(rc),
        "status": status,
        "wallclock_sec": float(ended - started),
        "log_path": str(log_path),
    }


def _extract_metrics(summary_path: Path, prefixes: Sequence[int]) -> Dict[str, Any]:
    if not summary_path.exists():
        out: Dict[str, Any] = {}
        for h in prefixes:
            out[f"success_at_{int(h)}"] = float("nan")
            out[f"goal_query_coverage_at_{int(h)}"] = float("nan")
            out[f"goal_cell_coverage_at_{int(h)}"] = float("nan")
        return out
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    progress = summary.get("progress_last", {}) if isinstance(summary, dict) else {}
    out = {
        "train_steps_total": summary.get("train_steps_total", float("nan")),
        "dataset_transitions": summary.get("dataset_transitions", float("nan")),
        "summary_path": str(summary_path),
    }
    for h in prefixes:
        h_int = int(h)
        out[f"success_at_{h_int}"] = float(progress.get(f"rollout_goal_success_rate_h{h_int}", np.nan))
        out[f"goal_query_coverage_at_{h_int}"] = float(progress.get(f"rollout_goal_query_coverage_rate_h{h_int}", np.nan))
        out[f"goal_cell_coverage_at_{h_int}"] = float(progress.get(f"rollout_goal_cell_coverage_rate_h{h_int}", np.nan))
    return out


def _write_markdown_summary(df: pd.DataFrame, out_path: Path, prefixes: Sequence[int]) -> None:
    lines: List[str] = []
    lines.append("# Swap Matrix Results")
    lines.append("")
    if df.empty:
        lines.append("No runs recorded.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    success_col = f"success_at_{int(prefixes[-1])}"
    ok_df = df[df["rc"] == 0].copy()
    lines.append(f"Rows: {len(df)}; successful rows: {len(ok_df)}")
    lines.append("")

    if not ok_df.empty:
        grouped = (
            ok_df.groupby(["mode", "collector", "learner"], dropna=False)[success_col]
            .agg(["count", "mean", "std"])
            .reset_index()
            .sort_values(["mode", "collector", "learner"])
        )
        lines.append("## Mean/Std By Cell")
        lines.append("")
        lines.append("| mode | collector | learner | n | mean_success@{} | std |".format(int(prefixes[-1])))
        lines.append("|---|---|---|---:|---:|---:|")
        for _, r in grouped.iterrows():
            lines.append(
                "| {mode} | {collector} | {learner} | {n} | {mean:.4f} | {std:.4f} |".format(
                    mode=r["mode"],
                    collector=r["collector"],
                    learner=r["learner"],
                    n=int(r["count"]),
                    mean=float(r["mean"]),
                    std=float(0.0 if pd.isna(r["std"]) else r["std"]),
                )
            )
        lines.append("")

        best_idx = grouped["mean"].idxmax()
        worst_idx = grouped["mean"].idxmin()
        best = grouped.loc[best_idx]
        worst = grouped.loc[worst_idx]
        lines.append(
            "Best cell: mode={mode}, collector={collector}, learner={learner}, mean_success@{h}={v:.4f}".format(
                mode=best["mode"],
                collector=best["collector"],
                learner=best["learner"],
                h=int(prefixes[-1]),
                v=float(best["mean"]),
            )
        )
        lines.append(
            "Worst cell: mode={mode}, collector={collector}, learner={learner}, mean_success@{h}={v:.4f}".format(
                mode=worst["mode"],
                collector=worst["collector"],
                learner=worst["learner"],
                h=int(prefixes[-1]),
                v=float(worst["mean"]),
            )
        )
        lines.append("")

    fail = (
        df.assign(failed=df["rc"] != 0)
        .groupby(["mode", "collector", "learner"], dropna=False)["failed"]
        .sum()
        .reset_index()
        .sort_values(["mode", "collector", "learner"])
    )
    lines.append("## Failure Counts")
    lines.append("")
    lines.append("| mode | collector | learner | failures |")
    lines.append("|---|---|---|---:|")
    for _, r in fail.iterrows():
        lines.append(
            "| {mode} | {collector} | {learner} | {f} |".format(
                mode=r["mode"], collector=r["collector"], learner=r["learner"], f=int(r["failed"])
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[1]
    python_default = _default_python(root_default)
    ap = argparse.ArgumentParser(description="Run collector-learner swap matrix for Maze2D diffuser vs SAC+HER sparse.")
    ap.add_argument("--root", type=Path, default=root_default)
    ap.add_argument("--python", type=Path, default=python_default)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--collectors", type=str, default=",".join(DEFAULT_COLLECTORS))
    ap.add_argument("--learners", type=str, default=",".join(DEFAULT_LEARNERS))
    ap.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES), help="warmstart,frozen")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--timeout-min", type=float, default=720.0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="Tiny fast run for pipeline validation.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    py = args.python if args.python.is_absolute() else (Path.cwd() / args.python)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")
    if not py.exists():
        raise SystemExit(f"Missing python: {py}")

    seeds = _parse_csv_int(args.seeds)
    collectors = _parse_csv(args.collectors)
    learners = _parse_csv(args.learners)
    modes = _parse_csv(args.modes)
    for c in collectors:
        _ = _method_script(root, c)
    for l in learners:
        _ = _method_script(root, l)
    for m in modes:
        if m not in {"warmstart", "frozen"}:
            raise SystemExit(f"Unsupported mode: {m}")

    base_dir = args.base_dir.resolve() if args.base_dir else (root / "runs" / "analysis" / "swap_matrix" / f"swap_matrix_{_stamp()}")
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "collector_replays").mkdir(parents=True, exist_ok=True)

    prefixes = (8, 16, 32) if args.smoke else (64, 128, 192, 256)

    common: Dict[str, Any] = {
        "env": "maze2d-umaze-v1",
        "device": "cpu" if args.smoke else args.device,
        "n_episodes": 4 if args.smoke else 400,
        "episode_len": 32 if args.smoke else 256,
        "horizon": 16 if args.smoke else 64,
        "max_path_length": 32 if args.smoke else 256,
        "train_steps": 20 if args.smoke else 6000,
        "batch_size": 16 if args.smoke else 128,
        "online_self_improve": True,
        "online_rounds": 1 if args.smoke else 4,
        "online_collect_episodes_per_round": 1 if args.smoke else 64,
        "online_collect_episode_len": 32 if args.smoke else 256,
        "online_train_steps_per_round": 10 if args.smoke else 3000,
        "online_replan_every_n_steps": 4 if args.smoke else 16,
        "online_goal_geom_p": 0.08 if args.smoke else 0.04,
        "online_goal_geom_min_k": 4 if args.smoke else 8,
        "online_goal_geom_max_k": 16 if args.smoke else 96,
        "online_goal_min_distance": 0.3 if args.smoke else 1.0,
        "online_early_terminate_threshold": 0.1 if args.smoke else 0.2,
        "online_min_accepted_episode_len": 8 if args.smoke else 64,
        "query_mode": "diverse",
        "num_eval_queries": 4 if args.smoke else 12,
        "query_bank_size": 32 if args.smoke else 256,
        "query_batch_size": 1,
        "query_min_distance": 0.3 if args.smoke else 1.0,
        "eval_goal_every": 10 if args.smoke else 3000,
        "goal_success_threshold": 0.2,
        "eval_rollout_horizon": 32 if args.smoke else 256,
        "eval_rollout_replan_every_n_steps": 4 if args.smoke else 16,
        # Avoid privileged maze-layout-aware candidate selection in default comparisons.
        "wall_aware_planning": False,
        "wall_aware_plan_samples": 1,
        "eval_success_prefix_horizons": ",".join(str(x) for x in prefixes),
        "save_checkpoint_every": 0 if args.smoke else 5000,
    }
    diffuser_only: Dict[str, Any] = {
        "n_diffusion_steps": 16 if args.smoke else 64,
        "model_dim": 32 if args.smoke else 64,
        "model_dim_mults": "1,2" if args.smoke else "1,2,4",
    }

    env = _base_env(root)
    timeout_sec = int(float(args.timeout_min) * 60.0)
    rows: List[Dict[str, Any]] = []

    def run_method(method: str, run_dir: Path, replay_import: Path | None, replay_export: Path | None, seed: int, mode: str) -> Dict[str, Any]:
        script = _method_script(root, method)
        cmd: List[str] = [str(py), str(script)]
        cfg = dict(common)
        if method == "diffuser":
            cfg.update(diffuser_only)
        cfg["seed"] = int(seed)
        cfg["logdir"] = str(run_dir)
        if replay_import is not None:
            cfg["replay_import_path"] = str(replay_import)
        if replay_export is not None:
            cfg["replay_export_path"] = str(replay_export)
        if mode == "frozen":
            cfg["disable_online_collection"] = True
        cmd.extend(_dict_to_cli(cfg))
        cmd.extend(_method_extra_args(method))

        log_path = run_dir / "stdout_stderr.log"
        if args.resume and (run_dir / "summary.json").exists() and (replay_export is None or replay_export.exists()):
            wallclock = 0.0
            rc = 0
            status = "skipped"
            result = {"rc": rc, "status": status, "wallclock_sec": wallclock, "log_path": str(log_path)}
        else:
            result = _run_cmd(cmd, cwd=root, env=env, log_path=log_path, timeout_sec=timeout_sec)
        return result

    # Phase 1: collector replay generation.
    replay_paths: Dict[tuple[str, int], Path] = {}
    for seed in seeds:
        for collector in collectors:
            run_dir = base_dir / "phase1_collectors" / collector / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            replay_path = base_dir / "collector_replays" / f"{collector}_seed{seed}.npz"
            replay_paths[(collector, seed)] = replay_path
            result = run_method(
                method=collector,
                run_dir=run_dir,
                replay_import=None,
                replay_export=replay_path,
                seed=seed,
                mode="warmstart",
            )
            rows.append(
                {
                    "phase": "collection",
                    "seed": int(seed),
                    "collector": collector,
                    "learner": collector,
                    "mode": "collector",
                    "run_dir": str(run_dir),
                    "replay_path": str(replay_path),
                    **result,
                }
            )

    # Phase 2: learner runs with imported collector replay.
    for seed in seeds:
        for collector in collectors:
            replay_path = replay_paths[(collector, seed)]
            for learner in learners:
                for mode in modes:
                    run_dir = base_dir / "phase2_learners" / mode / f"{collector}_to_{learner}" / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    result = run_method(
                        method=learner,
                        run_dir=run_dir,
                        replay_import=replay_path,
                        replay_export=None,
                        seed=seed,
                        mode=mode,
                    )
                    summary_path = run_dir / "summary.json"
                    metrics = _extract_metrics(summary_path=summary_path, prefixes=prefixes)
                    row = {
                        "phase": "learning",
                        "seed": int(seed),
                        "collector": collector,
                        "learner": learner,
                        "mode": mode,
                        "run_dir": str(run_dir),
                        "replay_path": str(replay_path),
                        **result,
                        **metrics,
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = base_dir / "swap_matrix_results.csv"
    md_path = base_dir / "swap_matrix_results.md"
    df.to_csv(csv_path, index=False)

    learn_df = df[df["phase"] == "learning"].copy() if not df.empty else pd.DataFrame()
    _write_markdown_summary(learn_df, out_path=md_path, prefixes=prefixes)

    print(f"[done] base_dir={base_dir}")
    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
