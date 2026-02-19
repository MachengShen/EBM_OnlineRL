# 2026-02-18 Maze2D Bootstrapping Ablation Plan

Goal: test the "collector-driven bootstrapping" hypothesis for online Maze2D Diffuser training by isolating:
1. collector vs learner weight usage (EMA vs online; teacher collector),
2. stochasticity (multi-seed),
3. replay nonstationarity (fixed replay snapshot + replay load).

This plan is executed via the existing runner:
- `scripts/synthetic_maze2d_diffuser_probe.py`

## Hypothesis (non-trivial)
"Diffuser's edge is primarily collector-driven via a planner-driven bootstrapping loop: early planning success yields higher-quality online replay, which then compounds."

## Environment Preconditions
The runner imports `d4rl` and may touch Mujoco even for `--help`. Use these env vars:

```bash
export ROOT=/root/ebm-online-rl-prototype
export PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONPATH="${ROOT}/third_party/diffuser-maze2d"
```

## Run Root Convention
```bash
export RUN_GROUP="2026-02-18_bootstrap"
export RUNS_ROOT="${ROOT}/output/bootstrapping/${RUN_GROUP}/maze2d"
mkdir -p "${RUNS_ROOT}"
```

## Baseline Command Template
Replace the budget knobs as needed; always set `--logdir` per run:

```bash
SEED=0
ABLATION="baseline"
RUN_DIR="${RUNS_ROOT}/${ABLATION}/seed_${SEED}"
mkdir -p "${RUN_DIR}"

( set -x
  ${PY} "${ROOT}/scripts/synthetic_maze2d_diffuser_probe.py" \
    --logdir "${RUN_DIR}" \
    --env maze2d-umaze-v1 \
    --seed "${SEED}" \
    --device cuda:0 \
    --query_mode diverse \
    --num_eval_queries 12 \
    --query_batch_size 1 \
    --goal_success_threshold 0.2 \
    --eval_rollout_mode receding_horizon \
    --eval_rollout_replan_every_n_steps 8 \
    --eval_rollout_horizon 256 \
    --eval_success_prefix_horizons 64,128,192,256 \
    --train_steps 6000 \
    --n_episodes 1000 \
    --episode_len 256 \
    --max_path_length 256 \
    --horizon 64 \
    --online_self_improve \
    --online_rounds 4 \
    --online_train_steps_per_round 3000 \
    --online_collect_transition_budget_per_round 4096 \
    --online_collect_episode_len 256 \
    --online_replan_every_n_steps 8 \
    --online_goal_geom_p 0.08 \
    --online_goal_geom_min_k 8 \
    --online_goal_geom_max_k 96 \
    --online_goal_min_distance 0.5
) 2>&1 | tee "${RUN_DIR}/stdout_stderr.log"
```

## Ablations

### A) Collector/Learner Weights (EMA vs Online)
Motivation: isolate whether online replay quality depends on the collector's weights being lagged (EMA) vs fresh (online).

- A0 baseline (default): `--collector_weights ema --eval_weights ema`
- A1 collect with online: `--collector_weights online --eval_weights ema`
- A2 evaluate online (diagnostic): `--collector_weights ema --eval_weights online`

### B) Teacher Collector (Frozen Checkpoint)
Motivation: isolate collection policy from the learner by using a frozen collector checkpoint.

Run with:
- `--collector_ckpt_path <path/to/checkpoint_last.pt>`
- `--collector_ckpt_weights ema` (or `online` if desired)

### C) Fixed Replay (Nonstationarity Control)
Motivation: determine whether instability/failure is due to nonstationary online replay rather than the learner update code.

Two useful modes:
1) Snapshot-and-freeze within one run:
   - `--fixed_replay_snapshot_round 1` (freeze after round 1)
   - optionally `--fixed_replay_snapshot_npz <RUN_DIR>/replay_snapshot_round1.npz`
2) Train-only from a saved replay:
   - produce replay once: `--replay_save_npz <RUN_DIR>/replay.npz`
   - then run new seeds with `--replay_load_npz <same replay.npz>` and `--online_rounds 0`

## Stop Criteria (Log Scan)
Run periodically:

```bash
rg -n -i "(nan|inf|overflow|diverg|assert|traceback)" "${RUNS_ROOT}" -S
```

Primary metric source-of-truth:
- `summary.json` (final rollup)
- `progress_metrics.csv` (time-series)

Heuristics:
- stop if NaNs/tracebacks appear
- stop if `progress_metrics.csv` stops updating for long periods (hung run)

## Reporting
For each condition, report (per seed):
- `summary.json.progress_last.rollout_goal_success_rate` (and prefix horizons)
- `online_collection_last` replay/transitions and whether replay froze (`replay_frozen`)
- whether any divergence markers triggered

Summarize in a small matrix:
- condition x seeds x final success metrics x divergence count

