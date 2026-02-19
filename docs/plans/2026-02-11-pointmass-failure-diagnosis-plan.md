# PointMass Diffuser Failure Diagnosis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a lightweight, non-intrusive diagnostic workflow to isolate why goal-conditioned planner behavior is not outperforming random.

**Architecture:** Add one standalone analysis script that loads an existing checkpoint, runs short random/planner rollouts, computes state-action coverage metrics, and emits a JSON+PNG report. Include model architecture introspection and a side-by-side hyperparameter comparison against Diffuser Maze2D references.

**Tech Stack:** Python, NumPy, Matplotlib, PyTorch, existing `ebm_online_rl` modules.

---

### Task 1: Add offline diagnosis script

**Files:**
- Create: `scripts/diagnose_pointmass_failure.py`

**Step 1: Implement CLI and checkpoint loading**
- Inputs: `--checkpoint`, `--device`, `--n_episodes`, `--episode_len`, `--bins`, `--seed`, `--out_prefix`.
- Load checkpoint config and state dict.

**Step 2: Implement rollout collection**
- Collect trajectories under:
  - `random` policy
  - `planner` policy (`plan_action` with single-sample planning)
- Keep run size small (default 100 episodes each).

**Step 3: Implement coverage/statistics metrics**
- State coverage on 2D grid in `[-1,1]^2`.
- Action coverage on 2D grid in `[-action_limit, action_limit]^2`.
- State-action occupancy ratio via `histogramdd`.
- Goal-reaching summary (`final_dist`, `min_dist`, success@0.05/0.10/0.20).

**Step 4: Implement architecture + hyperparameter comparison output**
- Print/current:
  - base dim, dim mults, kernel size, horizon, diffusion steps, predict_epsilon
  - parameter count
- Compare against Diffuser Maze2D defaults/reference (`dim=32`, `dim_mults=(1,4,8)`, `kernel=5`, `predict_epsilon=False`, `horizon=128/256`, `n_diffusion_steps=64/256`).

**Step 5: Save artifacts**
- Save JSON metrics and PNG heatmaps to `runs/analysis/`.

### Task 2: Run lightweight diagnosis

**Files:**
- Use: `scripts/diagnose_pointmass_failure.py`

**Step 1: Execute with current best available checkpoint**
- Recommended command:
  - `.venv/bin/python scripts/diagnose_pointmass_failure.py --checkpoint runs/online_pointmass_goal_diffuser/gpu_long_seed0/checkpoints/step_4000.pt --device cuda:0 --n_episodes 80 --bins 24`

**Step 2: Record outputs and summarize findings**
- Report where coverage appears insufficient vs where architecture/hyperparameter mismatch is likely dominant.

### Task 3: Verify script health

**Files:**
- Verify: `scripts/diagnose_pointmass_failure.py`

**Step 1: Syntax check**
- Run: `.venv/bin/python -m py_compile scripts/diagnose_pointmass_failure.py`

**Step 2: Dry-run import check**
- Run: `.venv/bin/python scripts/diagnose_pointmass_failure.py --help`
