#!/usr/bin/env python3
"""Compute and launch a fair-budget Maze2D CRL run.

Fairness objective:
- Align environment transitions with existing baselines.
- Align total gradient descents with existing baselines.

This script derives default targets from baseline summary JSON files and then
maps them to CRL knobs used by `lp_contrastive.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import List


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_summary_paths(compare_root: Path) -> List[Path]:
    return [
        compare_root / "diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0" / "summary.json",
        compare_root / "sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0" / "summary.json",
        compare_root / "gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0_rerun_20260217-203446" / "summary.json",
    ]


def _episode_len_for_env(env_name: str, override: int) -> int:
    if override > 0:
        return override
    if env_name.startswith("maze2d"):
        return 256
    raise ValueError(
        f"No default episode length for env={env_name!r}; "
        "pass --episode-length explicitly."
    )


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_compare_root = (
        repo_root
        / "runs"
        / "analysis"
        / "synth_maze2d_diffuser_probe"
        / "compare_diffuser_vs_gcbc_20260217-180356"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--compare-root", type=Path, default=default_compare_root)
    p.add_argument("--summary-path", type=Path, action="append", default=[])

    p.add_argument("--env-name", type=str, default="maze2d_umaze")
    p.add_argument("--alg", type=str, default="contrastive_nce")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=2)
    p.add_argument("--num-actors", type=int, default=1)

    p.add_argument("--target-transitions", type=int, default=0)
    p.add_argument("--target-gradient-steps", type=int, default=0)
    p.add_argument("--episode-length", type=int, default=0)

    p.add_argument("--step-limiter-key", choices=["learner_steps", "actor_steps"], default="learner_steps")
    p.add_argument("--num-sgd-steps-per-step", type=int, default=1)
    p.add_argument("--samples-per-insert-tolerance-rate", type=float, default=0.1)
    p.add_argument("--min-replay-size", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.num_sgd_steps_per_step <= 0:
        raise ValueError("--num-sgd-steps-per-step must be >= 1.")

    summary_paths = args.summary_path or _default_summary_paths(args.compare_root)
    missing = [str(p) for p in summary_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary files: {missing}")

    summaries = [_load_json(p) for p in summary_paths]
    dataset_transitions = [int(s["dataset_transitions"]) for s in summaries]
    train_steps_total = [int(s["train_steps_total"]) for s in summaries]

    target_transitions = args.target_transitions or max(dataset_transitions)
    target_gradient_steps = args.target_gradient_steps or max(train_steps_total)

    episode_len = _episode_len_for_env(args.env_name, args.episode_length)
    target_episodes = math.ceil(target_transitions / episode_len)
    aligned_actor_steps = target_episodes * episode_len

    # total gradient descents ~= learner_steps * num_sgd_steps_per_step
    learner_steps_target = math.ceil(target_gradient_steps / args.num_sgd_steps_per_step)
    # replay rate-limiter target in trajectory-units.
    samples_per_insert = (
        target_gradient_steps
        / float(args.num_sgd_steps_per_step * target_episodes)
    )
    if samples_per_insert <= 0:
        raise ValueError(f"Computed non-positive samples_per_insert={samples_per_insert}")

    min_replay_traj = args.min_replay_size // episode_len
    if min_replay_traj < 1:
        raise ValueError(
            f"min_replay_size={args.min_replay_size} is too small for "
            f"episode_length={episode_len}; need at least one trajectory."
        )
    # Reverb constraint in SampleToInsertRatio:
    # error_buffer = min_replay_traj * samples_per_insert * tolerance_rate
    # must satisfy error_buffer >= 2 * max(1.0, samples_per_insert).
    min_tolerance_required = (
        2.0 * max(1.0, samples_per_insert)
        / float(min_replay_traj * samples_per_insert)
    )
    effective_tolerance_rate = max(
        args.samples_per_insert_tolerance_rate,
        min_tolerance_required + 1e-6,
    )

    max_number_of_steps = (
        learner_steps_target if args.step_limiter_key == "learner_steps" else aligned_actor_steps
    )

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (
        f"crl_{args.alg}_fair_t{aligned_actor_steps}"
        f"_g{target_gradient_steps}_sgd{args.num_sgd_steps_per_step}"
        f"_seed{args.seed}_{ts}"
    )
    run_dir = args.compare_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    budget = {
        "summary_paths": [str(p) for p in summary_paths],
        "target_transitions": target_transitions,
        "target_gradient_steps": target_gradient_steps,
        "episode_length": episode_len,
        "target_episodes": target_episodes,
        "aligned_actor_steps": aligned_actor_steps,
        "num_sgd_steps_per_step": args.num_sgd_steps_per_step,
        "learner_steps_target": learner_steps_target,
        "step_limiter_key": args.step_limiter_key,
        "samples_per_insert": samples_per_insert,
        "samples_per_insert_tolerance_rate_requested": args.samples_per_insert_tolerance_rate,
        "samples_per_insert_tolerance_rate_min_required": min_tolerance_required,
        "samples_per_insert_tolerance_rate": effective_tolerance_rate,
        "min_replay_size": args.min_replay_size,
        "batch_size": args.batch_size,
        "max_number_of_steps": max_number_of_steps,
        "run_dir": str(run_dir),
    }
    (run_dir / "fair_budget_config.json").write_text(
        json.dumps(budget, indent=2),
        encoding="utf-8",
    )

    crl_dir = repo_root / "third_party" / "google-research-contrastive_rl" / "contrastive_rl"
    run_log = run_dir / "run.log"
    cmd = [
        "python",
        "lp_contrastive.py",
        "--env_name",
        args.env_name,
        "--alg",
        args.alg,
        "--start_index",
        str(args.start_index),
        "--end_index",
        str(args.end_index),
        "--seed",
        str(args.seed),
        "--num_actors",
        str(args.num_actors),
        "--max_number_of_steps",
        str(max_number_of_steps),
        "--step_limiter_key",
        args.step_limiter_key,
        "--num_sgd_steps_per_step",
        str(args.num_sgd_steps_per_step),
        "--samples_per_insert",
        f"{samples_per_insert:.8f}",
        "--samples_per_insert_tolerance_rate",
        str(effective_tolerance_rate),
        "--min_replay_size",
        str(args.min_replay_size),
        "--batch_size",
        str(args.batch_size),
    ]
    cmd_str = " ".join(shlex.quote(x) for x in cmd)

    shell_cmd = (
        f"cd {shlex.quote(str(crl_dir))} && "
        "source /root/miniconda3/bin/activate contrastive_rl && "
        "export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 && "
        "export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/root/miniconda3/envs/contrastive_rl/lib:$LD_LIBRARY_PATH && "
        "export D4RL_SUPPRESS_IMPORT_ERROR=1 && "
        f"{cmd_str} > {shlex.quote(str(run_log))} 2>&1"
    )

    (run_dir / "command.sh").write_text(shell_cmd + "\n", encoding="utf-8")
    print(json.dumps(budget, indent=2))
    print("\nLaunch command:")
    print(shell_cmd)

    if args.execute and not args.dry_run:
        env = os.environ.copy()
        proc = subprocess.run(["bash", "-lc", shell_cmd], env=env)
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
