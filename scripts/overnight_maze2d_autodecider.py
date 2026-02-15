#!/usr/bin/env python3
from __future__ import annotations

"""
Aggressive (but auditable) overnight auto-decider for Maze2D online Diffuser experiments.

Goal:
- Keep launching experiments without manual prompting.
- Adaptively explore/exploit hyperparameters based on fresh results.
- Allow variable "budget" per trial (train steps / online rounds), per user request.

Key design choices:
- We treat the online loop hyperparameters as the primary levers:
  - online_collect_episodes_per_round (data chunk size)
  - online_train_steps_per_round (update intensity)
  - online_replan_every_n_steps (control feedback cadence)
  - online_goal_geom_p (goal sampling geometry)
- We treat compute/budget knobs as a *fidelity dimension*:
  - train_steps (offline_init)
  - online_rounds
- We keep the policy fixed, but let the controller pick the next config
  based on observed outcomes (explore -> rank param influence -> promote best).
"""

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(v: Any) -> float:
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _log(fp: Path, msg: str) -> None:
    line = f"[autodecider] {_now_ts()} {msg}\n"
    print(line, end="")
    _append_text(fp, line)


def _isfinite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _softmax(xs: Sequence[float], temperature: float) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


@dataclass(frozen=True)
class TrialConfig:
    train_steps: int
    online_rounds: int
    online_collect_episodes_per_round: int
    online_train_steps_per_round: int
    online_replan_every_n_steps: int
    online_goal_geom_p: float

    def to_args(self) -> Dict[str, Any]:
        return {
            "train_steps": int(self.train_steps),
            "online_rounds": int(self.online_rounds),
            "online_collect_episodes_per_round": int(self.online_collect_episodes_per_round),
            "online_train_steps_per_round": int(self.online_train_steps_per_round),
            "online_replan_every_n_steps": int(self.online_replan_every_n_steps),
            "online_goal_geom_p": float(self.online_goal_geom_p),
        }

    def base_key(self) -> Tuple[Any, ...]:
        # Excludes budget knobs; used for promotion logic.
        return (
            int(self.online_collect_episodes_per_round),
            int(self.online_train_steps_per_round),
            int(self.online_replan_every_n_steps),
            float(self.online_goal_geom_p),
        )

    def short_name(self) -> str:
        gp = int(round(self.online_goal_geom_p * 1000))
        return (
            f"ts{self.train_steps}_or{self.online_rounds}"
            f"_ep{self.online_collect_episodes_per_round}"
            f"_t{self.online_train_steps_per_round}"
            f"_rp{self.online_replan_every_n_steps}"
            f"_gp{gp:03d}"
        )


@dataclass
class SearchSpace:
    train_steps: List[int]
    online_rounds: List[int]
    online_collect_episodes_per_round: List[int]
    online_train_steps_per_round: List[int]
    online_replan_every_n_steps: List[int]
    online_goal_geom_p: List[float]

    def budget_levels(self) -> List[Tuple[int, int]]:
        # Ordered by increasing cost.
        levels = sorted(set((int(ts), int(or_)) for ts in self.train_steps for or_ in self.online_rounds))
        levels.sort(key=lambda t: (t[0], t[1]))
        return levels


def default_search_space() -> SearchSpace:
    # Conservative-but-wide defaults; the auto-decider will focus based on results.
    return SearchSpace(
        train_steps=[1000, 2000, 3000, 4000],
        online_rounds=[2, 4, 6, 8, 12],
        online_collect_episodes_per_round=[8, 16, 32, 64],
        online_train_steps_per_round=[250, 500, 750, 1000, 1500, 2000, 3000],
        online_replan_every_n_steps=[4, 8, 16],
        online_goal_geom_p=[0.04, 0.08, 0.12],
    )


def _trial_from_parts(
    train_steps: int,
    online_rounds: int,
    eps: int,
    tsteps: int,
    replan: int,
    goal_p: float,
) -> TrialConfig:
    return TrialConfig(
        train_steps=int(train_steps),
        online_rounds=int(online_rounds),
        online_collect_episodes_per_round=int(eps),
        online_train_steps_per_round=int(tsteps),
        online_replan_every_n_steps=int(replan),
        online_goal_geom_p=float(goal_p),
    )


def _read_progress_latest(progress_csv: Path) -> Dict[str, Any] | None:
    if not progress_csv.exists():
        return None
    try:
        df = pd.read_csv(progress_csv)
    except Exception:
        return None
    if len(df) == 0:
        return None
    row = df.iloc[-1].to_dict()
    return row


def _summarize_progress(progress_csv: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not progress_csv.exists():
        return out
    try:
        df = pd.read_csv(progress_csv)
    except Exception:
        return out
    if len(df) == 0:
        return out
    out["progress_rows"] = int(len(df))
    out["step_last"] = int(df["step"].iloc[-1]) if "step" in df.columns else int(-1)
    # Prefer horizon-prefixed success (new protocol), but do not assume fixed horizons.
    # We dynamically discover all available prefix columns so the controller works with
    # both real runs (64/128/192/256) and smoke tests (e.g. 8/16/32).
    h_cols: List[Tuple[int, str]] = []
    for c in df.columns:
        m = re.match(r"^rollout_goal_success_rate_h(\d+)$", str(c))
        if m:
            h = int(m.group(1))
            h_cols.append((h, c))
    h_cols.sort(key=lambda t: t[0])
    for h, col in h_cols:
        if col in df.columns and df[col].notna().any():
            out[f"succ_h{h}_last"] = float(df[col].dropna().iloc[-1])
            out[f"succ_h{h}_max"] = float(df[col].dropna().max())
    if h_cols:
        out["max_prefix_horizon"] = int(h_cols[-1][0])
    if "imagined_goal_success_rate" in df.columns and df["imagined_goal_success_rate"].notna().any():
        out["imagined_succ_last"] = float(df["imagined_goal_success_rate"].dropna().iloc[-1])
        out["imagined_succ_max"] = float(df["imagined_goal_success_rate"].dropna().max())
    return out


def _summarize_online(online_csv: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not online_csv.exists():
        return out
    try:
        df = pd.read_csv(online_csv)
    except Exception:
        return out
    if len(df) == 0:
        return out
    out["online_rows"] = int(len(df))
    out["round_last"] = int(df["round"].iloc[-1]) if "round" in df.columns else int(-1)
    out["replay_transitions_last"] = int(df["replay_transitions"].iloc[-1]) if "replay_transitions" in df.columns else int(-1)
    # Planning success signals (added earlier).
    for col in ("planning_success_rate_final_t010", "planning_success_rate_final_t020", "planning_success_rate_final_rel090"):
        if col in df.columns and df[col].notna().any():
            out[col] = float(df[col].dropna().iloc[-1])
    return out


def _objective_from_summaries(progress_summary: Mapping[str, Any]) -> float:
    # Primary objective: best observed success at the longest available prefix horizon.
    # This is robust to different eval horizon choices (e.g. smoke tests).
    h = int(progress_summary.get("max_prefix_horizon", 256))
    v = _safe_float(progress_summary.get(f"succ_h{h}_max", float("nan")))
    if not _isfinite(v):
        v = _safe_float(progress_summary.get(f"succ_h{h}_last", float("nan")))
    return float(v)


def _param_importance(df: pd.DataFrame, objective_col: str, param_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = df[df[objective_col].notna()].copy()
    if len(base) == 0:
        return pd.DataFrame(rows)

    for p in param_cols:
        if p not in base.columns:
            continue
        grp = base.groupby(p, dropna=False)[objective_col].mean()
        if len(grp) == 0:
            continue
        best_val = grp.idxmax()
        rows.append(
            {
                "param": p,
                "n_values_seen": int(len(grp)),
                "mean_range": float(grp.max() - grp.min()) if len(grp) >= 2 else 0.0,
                "best_value": best_val,
                "best_mean": float(grp.max()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["mean_range", "best_mean"], ascending=[False, False])


def _weighted_choice(rng: random.Random, values: Sequence[Any], weights: Sequence[float]) -> Any:
    # Python's random.choices exists, but keep explicit for auditability.
    if len(values) != len(weights):
        raise ValueError("values/weights length mismatch")
    s = float(sum(weights))
    if not _isfinite(s) or s <= 0:
        return rng.choice(list(values))
    t = rng.random() * s
    acc = 0.0
    for v, w in zip(values, weights):
        acc += float(w)
        if t <= acc:
            return v
    return values[-1]


def _sample_value_bandit(
    rng: random.Random,
    df: pd.DataFrame,
    objective_col: str,
    param: str,
    values: Sequence[Any],
    temperature: float,
    explore_prob: float,
) -> Any:
    if rng.random() < explore_prob or param not in df.columns or objective_col not in df.columns:
        return rng.choice(list(values))
    sub = df[df[objective_col].notna()]
    if len(sub) < 3:
        return rng.choice(list(values))
    means = sub.groupby(param, dropna=False)[objective_col].mean().to_dict()
    scores = [float(means.get(v, float("nan"))) for v in values]
    # If unseen values exist, give them a neutral score to allow exploration.
    finite_scores = [s for s in scores if _isfinite(s)]
    neutral = float(sum(finite_scores) / len(finite_scores)) if finite_scores else 0.0
    scores = [s if _isfinite(s) else neutral for s in scores]
    weights = _softmax(scores, temperature=temperature)
    return _weighted_choice(rng, values, weights)


def _select_next_trial(
    rng: random.Random,
    space: SearchSpace,
    completed: pd.DataFrame,
    tried_keys: set[Tuple[Any, ...]],
    promote_prob: float,
    bandit_temperature: float,
    bandit_explore_prob: float,
) -> TrialConfig:
    param_cols = [
        "online_collect_episodes_per_round",
        "online_train_steps_per_round",
        "online_replan_every_n_steps",
        "online_goal_geom_p",
    ]
    objective_col = "objective_succ_h256"

    # Promotion: pick best base config and increase (train_steps, online_rounds) if possible.
    if len(completed) > 0 and rng.random() < promote_prob:
        base_cols = ["base_key", "train_steps", "online_rounds", objective_col]
        if all(c in completed.columns for c in base_cols):
            best = (
                completed[completed[objective_col].notna()]
                .sort_values([objective_col, "train_steps", "online_rounds"], ascending=[False, True, True])
                .head(1)
            )
            if len(best) == 1:
                bk = best.iloc[0]["base_key"]
                ts_cur = int(best.iloc[0]["train_steps"])
                or_cur = int(best.iloc[0]["online_rounds"])
                levels = space.budget_levels()
                try:
                    idx = levels.index((ts_cur, or_cur))
                except ValueError:
                    idx = -1
                if 0 <= idx < len(levels) - 1:
                    ts_next, or_next = levels[idx + 1]
                    # Reconstruct base params from base_key.
                    ep, tsteps, replan, gp = bk
                    cand = _trial_from_parts(ts_next, or_next, ep, tsteps, replan, gp)
                    if cand.base_key() == bk and cand.to_args().values():
                        key = tuple(sorted(cand.to_args().items()))
                        if key not in tried_keys:
                            return cand

    # Otherwise: sample a new base config using bandit-weighted values.
    ep = _sample_value_bandit(
        rng,
        completed,
        objective_col,
        "online_collect_episodes_per_round",
        space.online_collect_episodes_per_round,
        temperature=bandit_temperature,
        explore_prob=bandit_explore_prob,
    )
    tsteps = _sample_value_bandit(
        rng,
        completed,
        objective_col,
        "online_train_steps_per_round",
        space.online_train_steps_per_round,
        temperature=bandit_temperature,
        explore_prob=bandit_explore_prob,
    )
    replan = _sample_value_bandit(
        rng,
        completed,
        objective_col,
        "online_replan_every_n_steps",
        space.online_replan_every_n_steps,
        temperature=bandit_temperature,
        explore_prob=bandit_explore_prob,
    )
    goal_p = _sample_value_bandit(
        rng,
        completed,
        objective_col,
        "online_goal_geom_p",
        space.online_goal_geom_p,
        temperature=bandit_temperature,
        explore_prob=bandit_explore_prob,
    )

    # Budget selection: start cheaper by default (but user allows freedom).
    train_steps = rng.choice(space.train_steps)
    online_rounds = rng.choice(space.online_rounds)
    # Mild bias toward smaller budgets early (when we know little).
    if len(completed) < 4:
        train_steps = min(space.train_steps)
        online_rounds = min(space.online_rounds)

    cand = _trial_from_parts(train_steps, online_rounds, ep, tsteps, replan, float(goal_p))
    return cand


def _cmd_from_args(main_py: Path, logdir: Path, common: Mapping[str, Any], trial: TrialConfig) -> List[str]:
    args: List[str] = [str(main_py), "--logdir", str(logdir)]
    # Common args.
    for k, v in common.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
            else:
                args.append(f"--no_{k}")
        else:
            args.extend([f"--{k}", str(v)])
    # Trial args.
    for k, v in trial.to_args().items():
        args.extend([f"--{k}", str(v)])
    return args


def _run_one_trial(
    *,
    py: Path,
    main_py: Path,
    root: Path,
    base_dir: Path,
    driver_log: Path,
    common_args: Mapping[str, Any],
    trial: TrialConfig,
    monitor_every_sec: int,
    env: Mapping[str, str],
    max_wallclock_sec: int | None,
) -> Dict[str, Any]:
    run_dir = base_dir / f"{_stamp()}_{trial.short_name()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run.log"
    config_path = run_dir / "trial_config.json"

    payload = {"trial": dataclasses.asdict(trial), "common_args": dict(common_args)}
    _write_json(config_path, payload)

    cmd = [str(py)] + _cmd_from_args(main_py, run_dir, common_args, trial)
    _write_json(run_dir / "cmd.json", {"cmd": cmd})

    _log(driver_log, f"launch run_dir={run_dir} trial={trial.short_name()}")
    t0 = time.time()

    with run_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=dict(env),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    last_report = 0.0
    last_progress_rows = 0
    progress_csv = run_dir / "progress_metrics.csv"
    online_csv = run_dir / "online_collection.csv"

    while True:
        rc = proc.poll()
        now = time.time()
        if max_wallclock_sec is not None and (now - t0) > max_wallclock_sec:
            _log(driver_log, f"timeout_kill run_dir={run_dir} max_wallclock_sec={max_wallclock_sec}")
            proc.terminate()
            time.sleep(5)
            proc.kill()
            rc = proc.wait(timeout=30)
            break

        if now - last_report >= float(monitor_every_sec):
            last_report = now
            ps = _summarize_progress(progress_csv)
            osum = _summarize_online(online_csv)
            pr = int(ps.get("progress_rows", 0))
            # Only log when we see a new progress row to keep logs readable.
            if pr != last_progress_rows and pr > 0:
                last_progress_rows = pr
                _log(
                    driver_log,
                    "snapshot "
                    f"run={run_dir.name} rows={pr} "
                    f"succ256_last={_safe_float(ps.get('succ_h256_last', float('nan'))):.3f} "
                    f"succ256_max={_safe_float(ps.get('succ_h256_max', float('nan'))):.3f} "
                    f"online_round={osum.get('round_last', -1)} "
                    f"replay_transitions={osum.get('replay_transitions_last', -1)}",
                )

        if rc is not None:
            break
        time.sleep(1)

    elapsed = time.time() - t0
    _log(driver_log, f"done run_dir={run_dir} rc={rc} elapsed_sec={int(elapsed)}")

    ps = _summarize_progress(progress_csv)
    osum = _summarize_online(online_csv)
    obj = _objective_from_summaries(ps)
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "rc": int(rc) if rc is not None else -999,
        "elapsed_sec": float(elapsed),
        **trial.to_args(),
        "base_key": trial.base_key(),
        **{f"progress_{k}": v for k, v in ps.items()},
        **{f"online_{k}": v for k, v in osum.items()},
        "objective_succ_h256": float(obj),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/root/ebm-online-rl-prototype"))
    ap.add_argument("--python", type=Path, default=Path("/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python"))
    ap.add_argument("--main-script", type=Path, default=Path("/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py"))
    ap.add_argument("--budget-hours", type=float, default=12.0)
    ap.add_argument("--monitor-every-sec", type=int, default=300)
    ap.add_argument("--max-trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a tiny, fast configuration for functional testing of the controller "
            "(small dataset, short horizons, tiny train steps). Not intended for real results."
        ),
    )
    ap.add_argument("--promote-prob", type=float, default=0.55, help="Probability of promoting best config to higher budget.")
    ap.add_argument("--bandit-temperature", type=float, default=0.35)
    ap.add_argument("--bandit-explore-prob", type=float, default=0.25)
    ap.add_argument("--per-trial-timeout-min", type=float, default=None, help="Optional wallclock timeout per trial.")
    args = ap.parse_args()

    root: Path = args.root
    py: Path = args.python
    main_py: Path = args.main_script
    if not root.exists():
        raise SystemExit(f"missing root: {root}")
    if not py.exists():
        raise SystemExit(f"missing python: {py}")
    if not main_py.exists():
        raise SystemExit(f"missing main script: {main_py}")

    space = default_search_space()
    if args.smoke:
        # Fast preset for end-to-end testing: ensure we can (a) launch runs,
        # (b) see intermediate progress_metrics/online_collection flushes,
        # (c) write results/importance tables, without spending real GPU hours.
        space = SearchSpace(
            train_steps=[20, 40],
            online_rounds=[1, 2],
            online_collect_episodes_per_round=[1, 2],
            online_train_steps_per_round=[10, 20],
            online_replan_every_n_steps=[4, 8],
            online_goal_geom_p=[0.08],
        )

    base_dir = root / "runs/analysis/synth_maze2d_diffuser_probe" / f"autodecider_{_stamp()}"
    base_dir.mkdir(parents=True, exist_ok=True)
    driver_log = base_dir / "autodecider.log"

    # Common args (fixed for comparability; the controller mainly tunes online loop knobs).
    common_args: Dict[str, Any] = {
        "device": "cuda:0",
        "n_episodes": 1000,
        "episode_len": 256,
        "max_path_length": 256,
        "horizon": 64,
        "n_diffusion_steps": 64,
        "model_dim": 64,
        "model_dim_mults": "1,2,4",
        "learning_rate": 2e-4,
        "batch_size": 128,
        "val_every": 500,
        "val_batches": 20,
        # Required for monitoring/decider loops.
        "eval_goal_every": 500,
        "save_checkpoint_every": 5000,
        "query_mode": "diverse",
        "num_eval_queries": 24,
        "query_batch_size": 6,
        "query_resample_each_eval": True,
        "online_self_improve": True,
        "online_collect_episode_len": 256,
        "online_goal_geom_min_k": 8,
        "online_goal_geom_max_k": 96,
        "online_goal_min_distance": 0.5,
        "eval_rollout_horizon": 256,
        "eval_success_prefix_horizons": "64,128,192,256",
        "online_planning_success_thresholds": "0.1,0.2",
        "online_planning_success_rel_reduction": 0.9,
    }
    if args.smoke:
        # Reduce compute drastically; keep the same code paths exercised.
        common_args.update(
            {
                "device": "cpu",
                "n_episodes": 4,
                "episode_len": 32,
                "max_path_length": 32,
                "horizon": 16,
                "n_diffusion_steps": 16,
                "model_dim": 32,
                "model_dim_mults": "1,2",
                "batch_size": 16,
                "val_every": 10,
                "val_batches": 1,
                "eval_goal_every": 10,
                "save_checkpoint_every": 0,
                "num_eval_queries": 4,
                "query_bank_size": 32,
                "query_batch_size": 1,
                "eval_rollout_horizon": 32,
                "eval_success_prefix_horizons": "8,16,32",
                "wall_aware_plan_samples": 2,
                "online_collect_episode_len": 32,
                "online_goal_geom_min_k": 4,
                "online_goal_geom_max_k": 16,
                "online_goal_min_distance": 0.3,
            }
        )

    # Env for MuJoCo + diffuser imports.
    ld_mj = "/root/.mujoco/mujoco210/bin"
    env = dict(os.environ)
    env.update(
        {
            "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", "") + ":" + ld_mj,
            "MUJOCO_GL": "egl",
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(root / "third_party/diffuser-maze2d"),
        }
    )

    # Persist controller config for auditability.
    _write_json(
        base_dir / "autodecider_config.json",
        {
            "created_at": _now_ts(),
            "space": dataclasses.asdict(space),
            "common_args": common_args,
            "budget_hours": float(args.budget_hours),
            "max_trials": int(args.max_trials),
            "promote_prob": float(args.promote_prob),
            "bandit_temperature": float(args.bandit_temperature),
            "bandit_explore_prob": float(args.bandit_explore_prob),
        },
    )

    _log(driver_log, f"start base_dir={base_dir} max_trials={args.max_trials} budget_hours={args.budget_hours}")

    rng = random.Random(int(args.seed))
    deadline = datetime.now() + timedelta(hours=float(args.budget_hours))
    results_path = base_dir / "autodecider_results.csv"
    importance_path = base_dir / "autodecider_importance.csv"

    completed_rows: List[Dict[str, Any]] = []
    tried_keys: set[Tuple[Any, ...]] = set()

    for trial_idx in range(int(args.max_trials)):
        if datetime.now() >= deadline:
            _log(driver_log, f"budget_exhausted trial_idx={trial_idx}")
            break

        completed_df = pd.DataFrame(completed_rows) if completed_rows else pd.DataFrame()
        trial = _select_next_trial(
            rng,
            space,
            completed_df,
            tried_keys,
            promote_prob=float(args.promote_prob),
            bandit_temperature=float(args.bandit_temperature),
            bandit_explore_prob=float(args.bandit_explore_prob),
        )
        key = tuple(sorted(trial.to_args().items()))
        if key in tried_keys:
            # Try a few random perturbations to avoid wasting a trial slot.
            for _ in range(50):
                trial = _select_next_trial(
                    rng,
                    space,
                    completed_df,
                    tried_keys,
                    promote_prob=0.0,  # force exploration if stuck
                    bandit_temperature=float(args.bandit_temperature),
                    bandit_explore_prob=1.0,
                )
                key = tuple(sorted(trial.to_args().items()))
                if key not in tried_keys:
                    break
        tried_keys.add(key)

        _log(driver_log, f"trial_select idx={trial_idx} config={trial.short_name()}")

        per_trial_timeout = None
        if args.per_trial_timeout_min is not None:
            per_trial_timeout = int(float(args.per_trial_timeout_min) * 60.0)

        row = _run_one_trial(
            py=py,
            main_py=main_py,
            root=root,
            base_dir=base_dir,
            driver_log=driver_log,
            common_args=common_args,
            trial=trial,
            monitor_every_sec=int(args.monitor_every_sec),
            env=env,
            max_wallclock_sec=per_trial_timeout,
        )
        completed_rows.append(row)

        df = pd.DataFrame(completed_rows)
        df.to_csv(results_path, index=False)

        imp = _param_importance(
            df,
            objective_col="objective_succ_h256",
            param_cols=[
                "train_steps",
                "online_rounds",
                "online_collect_episodes_per_round",
                "online_train_steps_per_round",
                "online_replan_every_n_steps",
                "online_goal_geom_p",
            ],
        )
        if len(imp) > 0:
            imp.to_csv(importance_path, index=False)
            top = imp.iloc[0].to_dict()
            _log(
                driver_log,
                "importance_top "
                f"param={top.get('param')} range={_safe_float(top.get('mean_range')):.4f} "
                f"best_value={top.get('best_value')} best_mean={_safe_float(top.get('best_mean')):.4f}",
            )

        # Also append a concise entry to the repository handoff logs so humans can follow along.
        try:
            wm = root / "docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md"
            hf = root / "HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt"
            succ = float(row.get("objective_succ_h256", float("nan")))
            _append_text(
                hf,
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] [autodecider] completed {row.get('run_name')} "
                f"succ_h256={succ:.4f} cfg={trial.short_name()}\n",
            )
            _append_text(
                wm,
                "\n"
                f"## {datetime.now().strftime('%Y-%m-%d %H:%M %Z%z')}\n"
                "### Auto-decider update\n"
                f"- run: `{row.get('run_dir')}`\n"
                f"- cfg: `{trial.short_name()}`\n"
                f"- objective (best succ@256): `{succ:.4f}`\n"
                f"- results table: `{results_path}`\n",
            )
        except Exception as e:
            _log(driver_log, f"warn failed to append working-memory/handoff: {e}")

    _log(driver_log, f"done results={results_path} importance={importance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
